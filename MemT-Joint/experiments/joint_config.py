from __future__ import annotations

import copy
import json
import math
import hashlib
from pathlib import Path


def merge(a, b):
    result = copy.deepcopy(a)
    for key, value in b.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load(path, ancestry=()):
    path = Path(path).resolve()
    if path in ancestry:
        raise ValueError('Cyclic configuration include')
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream)
    result = {}
    for parent in value.pop('extends', []):
        result = merge(result, load(path.parent / parent, ancestry + (path,)))
    return merge(result, value)


def validate(config, training=False):
    if config['backend'] not in ('hf', 'verl', 'fixture'):
        raise ValueError('backend must be hf, verl, or fixture')
    if training and config['backend'] == 'fixture':
        raise ValueError('Fixture generation is not on-policy RL. Use smoke, not train.')
    if config['model']['temperature'] <= 0:
        raise ValueError('RL sampling requires positive temperature')
    if config['experiment']['mode'] not in ('source_backed', 'memory_only'):
        raise ValueError('Unknown evidence protocol')
    if not set(config['experiment']['train_roles']) <= {'B', 'U', 'R'}:
        raise ValueError('Unknown train role')
    if training and not config['experiment']['train_roles']:
        raise ValueError('Frozen baseline uses evaluate, not train')
    weights = config['joint']['branch_weights']
    total = sum(v for role in weights.values() for v in role.values())
    if abs(total - 1) > 1e-8 or any(not math.isfinite(v) or v < 0 for role in weights.values() for v in role.values()):
        raise ValueError('Explicit role/branch weights must be nonnegative and sum to one')
    if config['joint']['loss_variant'] not in ('step', 'token'):
        raise ValueError('Unknown clipping variant')
    if config['joint']['local_evaluation'] not in ('current_only', 'current_or_continuation'):
        raise ValueError('Unknown Local evaluation horizon')
    for name in ('global_rollouts', 'retrieval_questions', 'global_questions', 'local_questions'):
        if config['joint'][name] < 1:
            raise ValueError(name + ' must be positive')
    if config['retrieval']['steps'] < 1 or config['memory']['summary_context_turns'] < 1:
        raise ValueError('Steps and chunk width must be positive')
    reward = config['reward']
    if not (0 <= reward['evidence'] <= 1 and 0 <= reward['retrieval_cost'] <= .05 and
            0 <= reward['memory_cost'] <= .05 and 0 <= reward['edit_cost'] <= .01 and reward['invalid'] <= -3):
        raise ValueError('Reward coefficients violate the validated failure-penalty bound')
    if config.get('window'):
        raise ValueError('Window prefix snapshots are disabled; use empty-state end-to-end evaluation')
    if config.get('short_episode') and config['short_episode'].get('blocks') != 8:
        raise ValueError('The calibrated short-episode protocol uses at most eight blocks')
    if config['runtime'].get('checkpoint_retention') == 'latest_best' and config['backend'] != 'verl':
        raise ValueError('Joint latest_best retention requires distributed model shards')
    if config.get('protocol'):
        contract(config)


def contract(config):
    """Only inference conditions enter the comparison identity, never train-role weights."""
    p = config.get('protocol')
    if not p:
        return None
    if p.get('name') != 'memt-execution-v2' or p.get('policy_inference') != 'vllm' or p.get('answer_inference') != 'vllm':
        raise ValueError('Unknown common inference protocol/backend stack')
    if p.get('answer_weights') != 'current_policy' or p.get('answer_interface') != 'finish_answer':
        raise ValueError('Common protocol must use the current policy with finish(answer)')
    if config['memory'].get('backend') != 'chroma':
        raise ValueError('Common protocol requires the Chroma memory backend')
    if p.get('failure_policy') != 'fallback_visible_only' or not config['evaluation'].get('fallback_answer'):
        raise ValueError('Common evaluation requires visible-evidence fallback plus strict metrics')
    if (config['retrieval']['max_prompt_tokens'], config['model']['max_new_tokens'],
        config['retrieval']['max_observation_tokens'], config['retrieval']['observation_tokens']) != (39936, 1024, 20480, 20480):
        raise ValueError('The agreed common protocol uses 39936 prompt / 1024 output / 20480 observation tokens')
    from methods.joint.prompts import ANSWER_RULES, retrieval_prompt
    prompt = retrieval_prompt({'question': '__protocol_contract__'}, config['experiment']['mode'],
                              config['memory']['retrieval_topk'], config['evaluation']['max_steps'])
    sha = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    tools = {name: hashlib.sha256((root / 'memory' / name).read_bytes()).hexdigest()
             for name in ('branch.py', 'formation.py', 'formation_tools.py', 'update.py', 'retrieval.py', 'protocol.py')}
    reference = json.loads((root / 'MEMT_EXECUTION_REFERENCE.json').read_text(encoding='utf-8'))
    return {'name': p['name'], 'tool_prompt': 'memt-original-verbatim-v2',
        'memt_reference': reference, 'memory_construction': 'empty_state_chronological',
        'execution_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                            ('methods/joint/retrieval.py', 'methods/joint/prompts.py', 'models/joint.py')},
        'baseline_reuse': False, 'role_tags': False,
        'memory_tool_implementation': tools,
        'tool_prompt_sha256': sha(prompt), 'answer_prompt_sha256': sha(ANSWER_RULES),
        'answer_prompt': 'memt-original-category-format', 'failure_policy': p['failure_policy'],
        'answer_interface': 'finish(answer)',
        'policy_inference': 'vllm==0.8.5.post1', 'answer_inference': 'vllm==0.8.5.post1',
        'vllm_engine': 'V0', 'dtype': config['model']['dtype'],
        'prefix_caching': False, 'enforce_eager': True, 'disable_custom_all_reduce': True, 'seed': config['seed'],
        'tensor_parallel': config['distributed']['tensor_parallel'],
        'prompt_tokens': config['retrieval']['max_prompt_tokens'], 'output_tokens': config['model']['max_new_tokens'],
        'observation_tokens': config['retrieval']['observation_tokens'], 'path_tokens': config['retrieval']['token_budget'],
        'path_tokens_hard_limit': False,
        'eval_steps': config['evaluation']['max_steps'], 'top_k': config['memory']['retrieval_topk'],
        'memory_protocol': 'paper-v2', 'evidence_mode': config['experiment']['mode'],
        'sampling': {'temperature': 0, 'top_p': 1., 'top_k': -1, 'min_p': 0.},
        'answer_weights': 'current_policy', 'answer_base': config['models']['base'],
        'memory_backend': config['memory'].get('backend', 'chroma'),
        'embedding': config['memory']['embedding_model']}


def variant(config, name):
    result = copy.deepcopy(config)
    result['experiment']['variant'] = name
    if name == 'M0':
        result['experiment']['train_roles'] = []
    elif name == 'M1':
        result['experiment']['train_roles'] = ['R']
        result['joint'].update(global_rollouts=1, builder_rollouts=0, updater_rollouts=0)
        result['joint']['branch_weights'] = {'B': {'global': 0., 'local': 0.}, 'U': {'global': 0., 'local': 0.}, 'R': {'mot': 1.}}
    elif name == 'M2':
        result['experiment']['train_roles'] = ['B', 'U']
        result['joint']['branch_weights'] = {'B': {'global': .25, 'local': .25}, 'U': {'global': .25, 'local': .25}, 'R': {'mot': 0.}}
    elif name == 'M3':
        result['joint'].update(builder_rollouts=0, updater_rollouts=0)
        result['joint']['branch_weights'] = {'B': {'global': 1/3, 'local': 0.}, 'U': {'global': 1/3, 'local': 0.}, 'R': {'mot': 1/3}}
    elif name in ('M4', 'M5'):
        pass
    else:
        raise ValueError('Unknown study variant; use M0..M5')
    return result
