"""Real four-rank ZeRO-3 save/restore against an uninterrupted tiny Qwen3 run."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .io import load_config, read_json, sha256, write_json, write_jsonl


def tensor_files(path):
    from safetensors import safe_open
    mapping = {}
    for file in Path(path).glob('*.safetensors'):
        with safe_open(str(file), framework='pt', device='cpu') as reader:
            for key in reader.keys():
                mapping[key] = file
    return mapping


def main():
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg, out = load_config(args.config), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(cfg['models']['base'], local_files_only=True)
    base = out / 'tiny-model'
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=32,
                         max_position_embeddings=1024, attention_dropout=0.0, tie_word_embeddings=True,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id))
    model.save_pretrained(base, safe_serialization=True)
    tokenizer.save_pretrained(base)
    del model
    operations = []
    for i in range(40):
        response = '<tool_call>' + json.dumps({'name': 'create_raw', 'arguments': {'turn_ids': [f'D1:{i}']}}) + '</tool_call>'
        operations.append({'op_id': f'tiny-{i}', 'sample_id': 'tiny', 'completion_id': f'tiny-{i}',
                           'messages': [{'role': 'user', 'content': f'Archive [D1:{i}]: Alice lives in Paris ({i}).'}],
                           'response': response, 'target_span': [0, len(response)]})
    selected = out / 'selected.jsonl'
    write_jsonl(selected, operations)
    tiny = copy.deepcopy(cfg)
    tiny['construction'].update(steps=4, save_every=2, checkpoint_policy='full', max_length=256,
                                learning_rate=0.001, warmup_ratio=0.0, full_determinism=True, audit_full_state=True)
    tiny['storage']['ledger'] = str(out / 'storage-ledger.json')
    config = out / 'config.json'
    write_json(config, tiny)
    train = out / 'training'
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
               '-m', 'cli', 'train-sft', '--config', str(config), '--model', str(base),
               '--selected', str(selected), '--output', str(train)]
    deterministic = dict(os.environ, FLASH_ATTENTION_DETERMINISTIC='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                         NCCL_ALGO='Ring', NCCL_PROTO='Simple', PYTHONHASHSEED='42')
    subprocess.run(command, check=True, env=deterministic)
    expected = out / 'uninterrupted-final'
    shutil.copytree(train / 'final', expected)
    shutil.copytree(train / 'update-audit', out / 'uninterrupted-audit')
    expected_proofs = [read_json(train / f'parameter-proof-rank-{rank}.json') for rank in range(4)]
    checkpoint = train / 'checkpoint-2'
    optimizer_files = list(checkpoint.rglob('*optim_states.pt'))
    if len(optimizer_files) != 4 or len(list(checkpoint.glob('rng_state_*.pth'))) != 4:
        raise ValueError('Missing rank optimizer/RNG state in real ZeRO checkpoint')
    checkpoint_hashes = {str(p.relative_to(checkpoint)): sha256(p) for p in checkpoint.rglob('*') if p.is_file()}
    counters = []
    schedulers = []
    for file in optimizer_files:
        value = torch.load(file, map_location='cpu', weights_only=False)
        state = value['optimizer_state_dict']['optimizer_state_dict']['state']
        steps = sorted({int(v['step']) for v in state.values() if 'step' in v})
        if steps != [2]:
            raise ValueError('Optimizer counter at checkpoint is not 2: ' + str(steps))
        counters.append(steps)
    for file in checkpoint.rglob('*model_states.pt'):
        value = torch.load(file, map_location='cpu', weights_only=False)
        if value.get('lr_scheduler'):
            schedulers.append(value['lr_scheduler'])
    if not schedulers:
        raise ValueError('No saved scheduler state')
    subprocess.run(command + ['--resume', str(checkpoint)], check=True, env=deterministic)
    restore_state_proofs = []
    for rank in range(4):
        expected_state = read_json(out / 'uninterrupted-audit' / f'rank-{rank}-step-3.json')
        restored_state = read_json(train / 'update-audit' / f'rank-{rank}-step-3.json')
        for key in ('fp32_master_sha256', 'optimizer_sha256', 'optimizer_groups', 'bf16_parameter_sha256', 'microbatches', 'scheduler'):
            if expected_state[key] != restored_state[key]:
                raise ValueError(f'Restore state/data mismatch before update: rank={rank}, field={key}')
        restore_state_proofs.append({'rank': rank, 'master_optimizer_parameters_data_scheduler_equal': True,
                                    'torch_rng_equal': expected_state['torch_rng_sha256'] == restored_state['torch_rng_sha256'],
                                    'cuda_rng_equal': expected_state['cuda_rng_sha256'] == restored_state['cuda_rng_sha256']})
        if not restore_state_proofs[-1]['torch_rng_equal'] or not restore_state_proofs[-1]['cuda_rng_equal']:
            raise ValueError(f'Restored RNG differs before the next update: rank={rank}')
    left, right = tensor_files(expected), tensor_files(train / 'final')
    if left.keys() != right.keys():
        raise ValueError('Restored model tensor names changed')
    maximum, exact = 0.0, True
    for key in left:
        with safe_open(str(left[key]), framework='pt', device='cpu') as a, safe_open(str(right[key]), framework='pt', device='cpu') as b:
            x, y = a.get_tensor(key).float(), b.get_tensor(key).float()
            exact &= torch.equal(x, y)
            maximum = max(maximum, (x-y).abs().max().item())
            if not torch.allclose(x, y, atol=1e-5, rtol=1e-3):
                raise ValueError('Resumed update differs from uninterrupted run: ' + key)
    proofs = [read_json(train / f'parameter-proof-rank-{rank}.json') for rank in range(4)]
    if not all(p['resumed_step'] == 2 and p['completed_step'] == 4 for p in proofs):
        raise ValueError('Not all four ranks restored and continued')
    if not all(p['cpu_adam_bias_cache_normalized'] for p in proofs):
        raise ValueError('CPUAdam hidden bias-cache normalization was not active')
    if not any(p['changed'] for p in expected_proofs):
        raise ValueError('Tiny model never changed')
    for relative, value in checkpoint_hashes.items():
        if sha256(checkpoint / relative) != value:
            raise ValueError('Resume modified its source checkpoint')
    write_json(out / 'VERIFIED.json', {'world_size': 4, 'checkpoint_step': 2, 'final_step': 4,
               'optimizer_counters': counters, 'scheduler_states': schedulers, 'rng_files': 4,
               'bitwise_equal': exact, 'maximum_absolute_error': maximum, 'rtol': 1e-3, 'atol': 1e-5,
               'deterministic_flash_attention': True, 'restore_state_proofs': restore_state_proofs,
               'cpu_adam_bias_cache_normalized': True,
               'four_rank_proofs': proofs, 'checkpoint_sha256': checkpoint_hashes,
               'verification_scope': 'tiny Qwen3 only; not a 4B full-state resume claim'})


if __name__ == '__main__':
    main()
