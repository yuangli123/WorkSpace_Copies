from pathlib import Path

from experiments.io import sha256
from experiments.joint import identity
from experiments.joint_config import load
from scripts.make_fixture import create


def test_training_identity_binds_verl_runtime_version(tmp_path):
    root = Path(__file__).resolve().parents[1]
    create(tmp_path / 'data')
    contract = identity(load(root / 'configs/fixture.json'), tmp_path / 'data/train')
    resource = 'third_party/Tree-GRPO/verl/version/version'
    assert (root / resource).read_text().strip() == '0.1'
    assert contract['code'][resource] == sha256(root / resource)
