"""Configuration composition and explicit model roles, without loading weights."""
from __future__ import annotations

import copy
import math
from pathlib import Path


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path, ancestors=()):
    import yaml
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError('Configuration include cycle: ' + str(path))
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a mapping')
    result = {}
    includes = value.pop('extends', [])
    if not isinstance(includes, list) or not all(isinstance(p, str) for p in includes):
        raise ValueError('extends must be a list of relative configuration paths')
    for parent in includes:
        result = merge(result, load_config(path.parent / parent, (*ancestors, path)))
    result = merge(result, value)
    return normalize(result)


def normalize(result):
    for section, names in {'model': ['temperature'], 'retrieval': ['learning_rate', 'warmup_ratio', 'kl_beta', 'clip', 'evidence_alpha'],
                           'construction': ['learning_rate', 'warmup_ratio', 'trace_lambda', 'keep_fraction', 'validation_fraction']}.items():
        for name in names:
            if result.get(section, {}).get(name) is not None:
                number = float(result[section][name])
                if not math.isfinite(number):
                    raise ValueError(section + '.' + name + ' must be finite')
                result[section][name] = number
    if 'models' in result:
        roles = result['models']
        if set(roles) != {'base', 'construction', 'retrieval', 'embedding', 'judge'}:
            raise ValueError('Specify all five model roles')
        if not all(isinstance(v, str) and v for v in roles.values()):
            raise ValueError('Model role paths must be nonempty strings')
        result.setdefault('memory', {})['embedding_model'] = roles['embedding']
    protocol = result.get('construction', {}).get('protocol', 'legacy-v1')
    if protocol not in ('legacy-v1', 'paper-v2'):
        raise ValueError('Unknown construction protocol')
    return result
