"""Exercise the launcher on the cluster's actual Bash, including nounset."""
import os
import subprocess
import re
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name == 'nt', reason='Validate with the real cluster Bash on Linux')
@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('launcher', ['slurm_memt_execution_50.sh', 'slurm_short8_50.sh'])
def test_train_launcher_arguments_with_nounset(tmp_path, resume, launcher):
    script = (Path(__file__).resolve().parents[1] / 'scripts' / launcher).read_text()
    start = script.index('set --\n')
    setup = script[start:script.index('python -u cli.py train', start)]
    if resume:
        (tmp_path/'train').mkdir()
        (tmp_path/'train/LATEST.json').write_text('{}')
    command = 'RUN=$1\n' + setup + '\nprintf "%s\\n" "$#"\nif (( $# )); then printf "%s\\n" "$1"; fi'
    result = subprocess.run(['/bin/bash', '-euc', command, 'argument-check', str(tmp_path)],
                            capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == (['1', '--resume'] if resume else ['0'])


def test_formal_config_binding_ignores_only_run_stop_path(tmp_path, monkeypatch):
    from experiments.joint_config import load
    from experiments.io import write_json
    root = Path(__file__).resolve().parents[1]
    script = (root/'scripts/slurm_short8_50.sh').read_text()
    code = next(chunk.split('\nPY\n')[0] for chunk in script.split("<<'PY'\n")
                if 'expected=receipt' in chunk.split('\nPY\n')[0])
    receipt_path = re.search(r"'([^']+/PROFILE_PASSED.json)'", code).group(1)
    profile = load(root/'configs/short8-profile.json')
    profile['runtime']['stop_file'] = '/previous-profile/train/STOP'
    common = dict(code={}, models={}, data={}, common_protocol={})
    receipt = dict(common, config=profile, scope='training_compute_checkpoint_separate')
    write_json(tmp_path/receipt_path, receipt)
    actual = dict(common, config=load(root/'configs/short8-m4-50.json'))
    file = tmp_path/'preflight.json'
    write_json(file, {'identity':actual})
    monkeypatch.setattr(sys, 'argv', ['verify',str(tmp_path),str(file)])
    exec(code, {})
    actual['config']['retrieval']['learning_rate'] *= 2
    write_json(file, {'identity':actual})
    with pytest.raises(AssertionError): exec(code, {})
