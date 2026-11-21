"""Run CPU checks without modifying an existing Python environment."""
import json
import os
from pathlib import Path
import platform
import sys
import uuid


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    # Import native libraries before pytest's Windows exception hooks.
    import torch
    import pytest
    root = Path(__file__).resolve().parents[1]
    temp = root / 'artifacts' / 'test-runs' / uuid.uuid4().hex
    temp.parent.mkdir(parents=True, exist_ok=True)
    results = []
    class Recorder:
        def pytest_runtest_logreport(self, report):
            if report.when == 'call' or report.failed:
                results.append({'test': report.nodeid, 'phase': report.when, 'outcome': report.outcome})
    args = ['-q', '-p', 'no:cacheprovider', '--basetemp', str(temp), str(root / 'tests')]
    if sys.platform == 'win32':
        args += ['-p', 'no:faulthandler']
    result = pytest.main(args, plugins=[Recorder()])
    target = root / 'artifacts' / 'validation' / 'LOCAL_CHECKS.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'exit_code': int(result), 'python': platform.python_version(), 'torch': torch.__version__,
        'device': 'cpu', 'results': results, 'scope': 'fixture environments, synthetic on-policy micro-model, CPU checkpoint equality',
        'not_verified': ['Qwen3-4B training', 'four-A100 FSDP/vLLM execution', 'real benchmark quality', 'M6 distributed SFT execution']},
        indent=2, ensure_ascii=False), encoding='utf-8')
    return int(result)


if __name__ == '__main__':
    raise SystemExit(main())
