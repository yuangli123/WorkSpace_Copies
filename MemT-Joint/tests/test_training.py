import copy
import math
from types import SimpleNamespace

import pytest
import torch

from experiments.joint_config import load
from methods.joint.loss import joint_loss
from training.native import NativeUpdater
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_step_ratio_is_not_token_clipping_and_detaches():
    current = torch.tensor([[math.log(2), math.log(.5), -1000.]], requires_grad=True)
    old = torch.zeros_like(current, requires_grad=True)
    ref = torch.zeros_like(current, requires_grad=True)
    a = torch.tensor([1.], requires_grad=True)
    mask, weight = torch.tensor([[1., 1., 0.]]), torch.tensor([1.])
    loss, _ = joint_loss(current, old, ref, a, mask, weight, beta=0)
    token, _ = joint_loss(current, old, ref, a, mask, weight, beta=0, variant='token')
    assert loss.item() == pytest.approx(-1.)
    assert token.item() == pytest.approx(-.85)
    loss.backward()
    assert current.grad[0, 2] == 0 and torch.isfinite(current.grad).all()
    assert old.grad is None and ref.grad is None and a.grad is None


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)

    def forward(self, input_ids, **kwargs):
        # Causal state depends on the complete sampled prefix.
        return SimpleNamespace(logits=self.head(self.embedding(input_ids).cumsum(1)))


def updater():
    cfg = load(ROOT / 'configs/hf-smoke.json')
    cfg['model']['gradient_checkpointing'] = False
    cfg['model']['temperature'] = .8
    policy = SimpleNamespace(model=Tiny(), device=torch.device('cpu'))
    return NativeUpdater(policy, cfg)


def rows():
    return [{'prompt_ids': [1, 2], 'response_ids': [3, 4], 'advantage': 1., 'weight': .5},
            {'prompt_ids': [1], 'response_ids': [5], 'advantage': -.7, 'weight': .5}]


def test_actual_parameter_update_and_exact_cpu_resume(tmp_path):
    torch.manual_seed(4)
    original = updater()
    before = copy.deepcopy(original.policy.model.state_dict())
    reference = copy.deepcopy(original.reference.state_dict())
    training_rows = rows()
    info = original.update(training_rows)
    assert info['grad_norm'] > 0
    assert any(not torch.equal(before[k], v) for k, v in original.policy.model.state_dict().items())
    assert all(torch.equal(reference[k], v) for k, v in original.reference.state_dict().items())
    assert all('old_log_probs' in r and 'ref_log_probs' in r for r in training_rows)
    original.save(tmp_path / 'checkpoint', {'step': 1})
    original.update(rows())
    resumed = updater()
    assert resumed.restore(tmp_path / 'checkpoint') == {'step': 1}
    resumed.update(rows())
    for key, value in original.policy.model.state_dict().items():
        assert torch.equal(value, resumed.policy.model.state_dict()[key])
    for a, b in zip(original.optimizer.state.values(), resumed.optimizer.state.values()):
        for key in a:
            assert torch.equal(a[key], b[key]) if torch.is_tensor(a[key]) else a[key] == b[key]
    (tmp_path / 'checkpoint/state.json').write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        resumed.restore(tmp_path / 'checkpoint')
