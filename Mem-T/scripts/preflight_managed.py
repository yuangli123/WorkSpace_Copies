"""Actual four-rank optimizer/save/restore/export check using synthetic tiny-model targets."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.io import configure_storage, load_config, read_json, sha256, write_json, write_jsonl
from memory.protocol import Completion
from training.verl_backend import VerlBackend
from experiments.workflow import train_retrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    root = Path('/public/home/wwxia/yuang.li/workspace/Mem-T')
    assert Path.cwd().resolve() == root
    output = Path(args.output).resolve()
    cfg = load_config(root / 'configs/memt4b_4gpu_100gb.yaml')
    cfg['retrieval'].update(steps=6, save_every=1, validate_every=2, batch_size=32, trees=1,
                            max_depth=2, pivots=1, max_prompt_tokens=64, max_observation_tokens=64,
                            learning_rate=0.001, warmup_ratio=0)
    cfg['model'].update(max_new_tokens=16)
    cfg['distributed'].update(max_batched_tokens=256, gpu_memory_utilization=0.15)
    cfg['runtime'].update(restore_probe_step=2, stop_file=str(output.parent / 'TINY_STOP'),
                          fixture_only=True, fixture_source_sha256=sha256(__file__))
    model = Path('/public/home/wwxia/yuang.li/models/memt-environment-smoke/qwen3-tiny')
    cfg['model']['checkpoint'] = str(model)
    configure_storage(cfg)
    questions = [{'qa_id': str(i), 'sample_id': 'fixture', 'question': 'city?', 'category': 1,
                  'answer': 'Paris', 'evidence': []} for i in range(32)]
    class Environment:
        def prompt(self, *unused):
            return [{'role': 'user', 'content': 'city?'}]
        def step(self, text, seen):
            return True, True, text, '', [], seen
    memories = []
    for split in ('train', 'valid'):
        folder = output.parent / 'tiny-fixture-data' / split
        write_json(folder / 'index.json', {'fixture': True})
        write_jsonl(folder / 'questions.jsonl', questions)
        memories.append(SimpleNamespace(path=folder, partition={'split': split}, questions=questions,
                                        environment=lambda *unused: Environment()))
    backend = VerlBackend(model, cfg)
    class FixturePolicy:
        def complete_batch(self, messages, seeds, **unused):
            return [self.completion('Paris' if seed % 2 else 'Rome') for seed in seeds]
        def completion(self, text):
            return Completion(text, backend.tokenizer.encode('city?', add_special_tokens=False),
                              backend.tokenizer.encode(text, add_special_tokens=False))
        def complete(self, messages, **unused):
            return self.completion('Paris')
        def check_observation(self, text):
            return backend.check_observation(text)
    assert FixturePolicy().check_observation('') is None
    train_retrieval(cfg, memories[0], FixturePolicy(), backend, output, memories[1], args.resume)
    if (output / 'TRAINING_COMPLETE.json').exists():
        import torch
        from transformers import AutoModelForCausalLM
        before = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).state_dict()
        after = AutoModelForCausalLM.from_pretrained(output / 'final', torch_dtype=torch.bfloat16).state_dict()
        assert before.keys() == after.keys()
        assert any(not torch.equal(value, after[key]) for key, value in before.items())
        assert all(torch.isfinite(value).all() for value in after.values())
        assert read_json(output / 'RESTORE_VERIFIED.json')['model_optimizer_scheduler_rng_equal']
        report = {'fixture_only': True, 'world_size': 4, 'global_query_batch': 32, 'steps': 6,
                  'actual_parameters_changed': True, 'restore_verified': True,
                  'best': read_json(output / 'BEST.json')}
        write_json(output / 'PREFLIGHT_PASSED.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
