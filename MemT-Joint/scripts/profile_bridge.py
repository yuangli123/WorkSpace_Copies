"""Disposable GPU check of the native/separate-policy vLLM weight bridge."""
import argparse
import json
import traceback

from experiments.io import checked_output, write_json
from experiments.joint_config import load, validate


def read_embedding(worker):
    return float(dict(worker.get_model().named_parameters())['model.embed_tokens.weight'][0, 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = load('configs/common-m5.json')
    validate(config)
    output = checked_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'capacity_only': True, 'formal_updates': 0}
    try:
        import torch
        assert torch.cuda.device_count() == 2
        from models.hf import HFPolicy
        from training.separate import RoleBundle
        config['model'].update(device='cpu', cpu_dtype='bfloat16')
        owner = HFPolicy(config['models']['base'], config)
        alternate = HFPolicy(config['models']['base'], config)
        bundle = RoleBundle({'B':owner, 'U':alternate, 'R':alternate}, 'cuda:0')
        bundle.common_config = config
        policy = bundle.generation_for('B')
        messages = [{'role': 'user', 'content': 'Reply with OK.'}]
        first = policy.complete(messages, max_new_tokens=8, temperature=0, stop_at_tool=False)
        with policy.phase():
            before = policy.engine.collective_rpc(read_embedding)
        with torch.no_grad():
            owner.model.get_input_embeddings().weight[0, 0].add_(1)
        expected = float(owner.model.get_input_embeddings().weight[0, 0])
        policy.invalidate()
        with policy.phase():
            after = policy.engine.collective_rpc(read_embedding)
        assert abs(after[0]-expected) < .01 and abs(after[0]-before[0]) > .9, (before, after, expected)
        assert bundle.generation_for('U') is policy
        with policy.phase():
            other = policy.engine.collective_rpc(read_embedding)
        assert abs(other[0]-before[0]) < .01, (other,before)
        assert bundle.generation_for('B') is policy
        second = policy.complete(messages, max_new_tokens=8, temperature=0, stop_at_tool=False)
        assert first.response_ids and second.response_ids
        report.update(passed=True, before=before, after=after, expected_rank0=expected,
                      alternate_owner=other, single_engine_role_switch=True,
                      generated_lengths=[len(first.response_ids),len(second.response_ids)])
        policy.close()
        report['closed_cleanly'] = True
        write_json(output / 'BRIDGE_PASSED.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
        write_json(output / 'BRIDGE_FAILED.json', report)
        raise


if __name__ == '__main__':
    main()
