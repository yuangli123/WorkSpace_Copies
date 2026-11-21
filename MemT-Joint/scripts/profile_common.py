"""Disposable capacity probe, never counted as an on-policy training update."""
import argparse
import json
import time
import traceback
from pathlib import Path

from experiments.io import write_json, checked_output
from experiments.joint_config import load, validate, contract
from experiments.runtime import ResourceMonitor


def long_messages(tokenizer, limit):
    from models.hf import render_prompt
    def messages(n):
        return [{'role': 'system', 'content': 'Read the evidence and answer with one short phrase.'},
                {'role': 'user', 'content': ' evidence' * n + '\nWhat is provided?'}]
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        size = len(tokenizer.encode(render_prompt(tokenizer, messages(mid)), add_special_tokens=False))
        if size <= limit:
            lo = mid
        else:
            hi = mid - 1
    value = messages(lo)
    size = len(tokenizer.encode(render_prompt(tokenizer, value), add_special_tokens=False))
    assert limit - 8 <= size <= limit, size
    return value, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/common-a100x2-50.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = load(args.config)
    validate(config, training=True)
    output = checked_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'capacity_only': True, 'formal_updates': 0, 'contract': contract(config), 'stages': []}
    started = time.time()
    def stage(name, **values):
        report['stage'] = name
        report['elapsed_seconds'] = time.time() - started
        report['stages'].append(dict(name=name, **values))
        write_json(output / 'progress.json', report)
        print('CAPACITY_STAGE ' + json.dumps(report['stages'][-1]), flush=True)
    with ResourceMonitor(output, config):
        try:
            import torch
            assert torch.cuda.device_count() == 2
            stage('initializing')
            from experiments.joint import make_runtime
            runtime, _, _, updater = make_runtime(config, training=True)
            policy = runtime.policy
            messages, prompt_tokens = long_messages(policy.tokenizer, policy.prompt_limit)
            stage('vllm_prefill', prompt_tokens=prompt_tokens, requested_output_tokens=1024)
            result = policy.complete(messages, temperature=0, stop_at_tool=False)
            stage('vllm_prefill_passed', actual_output_tokens=len(result.response_ids))
            stage('current_policy_answer_prefill', prompt_tokens=prompt_tokens, requested_output_tokens=1024)
            answer = policy.complete(messages, temperature=0, stop_at_tool=False)
            reserved = torch.cuda.memory_reserved(0)
            stage('current_policy_answer_passed', actual_output_tokens=len(answer.response_ids),
                  driver_reserved_mib=reserved/2**20, driver_allocated_mib=torch.cuda.memory_allocated(0)/2**20)
            # Full boundary allocation, including old/reference probabilities and Adam state.
            # These synthetic targets are discarded; they are not an RL effectiveness result.
            response = policy.tokenizer.encode(' evidence', add_special_tokens=False)[0]
            rows = [dict(role=role, prompt_ids=result.prompt_ids, response_ids=[response]*1024,
                         advantage=adv, weight=.5) for role, adv in [('B', 1.), ('R', -.7)]]
            stage('full_length_update', batch=2, response_tokens=1024)
            metrics = updater.update(rows)
            assert metrics['actor/grad_norm'] > 0, metrics
            assert abs(metrics['actor/loss'] + .15) < .03, metrics
            stage('full_length_update_passed', metrics=metrics)
            stage('warm_optimizer_update', batch=2, response_tokens=1024)
            metrics = updater.update(rows)
            assert metrics['actor/grad_norm'] > 0, metrics
            stage('warm_optimizer_update_passed', metrics=metrics)
            report['passed'] = True
            write_json(output / 'CAPACITY_PASSED.json', report)
        except BaseException as error:
            report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
            write_json(output / 'CAPACITY_FAILED.json', report)
            raise


if __name__ == '__main__':
    main()
