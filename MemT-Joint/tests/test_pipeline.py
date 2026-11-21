"""Real CPU autograd through the runner, using a tiny sampled tool-token policy.

The token vocabulary encodes synthetic tool actions; it is not a Qwen result.
"""
import copy
import json
import re
from contextlib import contextmanager
from pathlib import Path

import torch

from experiments.joint_config import load
from experiments.joint import run, no_phase
from models.joint import FixturePolicy, Runtime
from memory.branch import Ranker
from memory.protocol import Completion
from scripts.make_fixture import create
from tests.test_training import Tiny
from training.native import NativeUpdater

ROOT = Path(__file__).resolve().parents[1]


class MicroPolicy:
    def __init__(self, weights):
        self.model, self.device = Tiny(), torch.device('cpu')
        self.model.load_state_dict(weights)
        self.role = 'R'

    def generation_for(self, role):
        self.role = role
        return self

    def set_seed(self, seed):
        torch.manual_seed(seed)

    def complete(self, messages, temperature=None, **kwargs):
        role = self.role
        prompt = [{'B': 1, 'U': 2, 'R': 3}[role], len(messages) % 7]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor([prompt])).logits[0, -1]
            action = int(logits.argmax()) if temperature == 0 else int(torch.multinomial((logits / temperature).softmax(-1), 1))
        def tool(name, **args):
            return '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'
        user = messages[-1]['content']
        if role == 'B':
            sources = re.findall(r'\[([^\]\n]+)\][^\n]*?residence=(\w+)', user)
            text = tool('update_summary', content=';'.join('residence=' + c for _, c in sources))
            if action % 2:
                text += ''.join(tool('create_fact', fact='residence=' + c, start_time='', end_time='') for _, c in sources)
            if action % 3 == 0:
                text += tool('create_raw', turn_ids=[sid for sid, _ in sources])
        elif role == 'U':
            match = re.search(r'\[New memory to be added\]\s*(.*?)\s*\[Existing memories', user, re.S)
            item = json.loads(match.group(1))
            text = tool('add_item', document=item['document']) if action % 2 else tool('ignore_item', reason='micro action')
        else:
            observation = user.split('<observation>')[1].split('</observation>')[0] if '<observation>' in user else ''
            if observation and action % 2 == 0:
                values = re.findall(r'residence=(\w+)', observation)
                text = tool('finish', answer=values[-1] if values else 'unknown')
            else:
                text = tool(['search_facts', 'search_summary', 'search_turns'][action % 3], query='residence')
        return Completion(text, prompt, [action])


def test_training_validation_interruption_and_resume(tmp_path, monkeypatch):
    create(tmp_path / 'data')
    cfg = load(ROOT / 'configs/fixture.json')
    cfg.update(backend='hf')
    cfg['model'].update(device='cpu', answer_device='cpu', dtype='float32', gradient_checkpointing=False)
    cfg['memory']['backend'] = 'lexical_test'
    cfg['retrieval'].update(steps=2, warmup_ratio=0.0)
    cfg['runtime'].update(save_every=1, validate_every=1, monitor_interval=0)
    base, embedding = tmp_path / 'base', tmp_path / 'embedding'
    base.mkdir(); embedding.mkdir()
    torch.manual_seed(9)
    initial = copy.deepcopy(Tiny().state_dict())
    torch.save(initial, base / 'model.bin')
    (embedding / 'config.json').write_text('{}')
    cfg['models']['base'] = str(base)
    cfg['memory']['embedding_model'] = str(embedding)
    output = tmp_path / 'run'
    first = [True]

    def factory(config, training):
        policy = MicroPolicy(initial)
        fixed = FixturePolicy()
        runtime = Runtime(policy, config, fixed=fixed, ranker=Ranker(fixture=True))
        updater = NativeUpdater(policy, config)
        update = updater.update
        def stop_after_first(rows):
            values = update(rows)
            if first[0]:
                (output / 'STOP').write_text('test interruption')
                first[0] = False
            return values
        updater.update = stop_after_first
        return runtime, fixed, no_phase, updater

    monkeypatch.setattr('experiments.joint.make_runtime', factory)
    run(cfg, tmp_path / 'data/train', output, valid_data=tmp_path / 'data/valid')
    assert json.loads((output / 'LATEST.json').read_text())['step'] == 1
    assert not (output / 'TRAINING_COMPLETE.json').exists()
    (output / 'STOP').unlink()
    run(cfg, tmp_path / 'data/train', output, resume=True, valid_data=tmp_path / 'data/valid')
    assert json.loads((output / 'TRAINING_COMPLETE.json').read_text())['step'] == 2
    best = json.loads((output / 'BEST.json').read_text())
    assert best['step'] in (1, 2)
    assert (output / best['validation'] / 'COMPLETE.json').exists()
    # A completed resume is idempotent, including pending validation checks.
    run(cfg, tmp_path / 'data/train', output, resume=True, valid_data=tmp_path / 'data/valid')
    assert json.loads((output / 'LATEST.json').read_text())['step'] == 2
