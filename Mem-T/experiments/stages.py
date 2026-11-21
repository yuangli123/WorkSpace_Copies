"""Single-run ownership, immutable stage contracts, and verified stage reuse."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

from .io import manifest, read_json, sha256, write_json
from .storage import DiskBudget


class StageDeferred(RuntimeError):
    """The allocation is ending; sealed records may continue in the next job."""


def remaining_seconds():
    end = os.environ.get('SLURM_JOB_END_TIME')
    return float(end) - time.time() if end else float('inf')


def stop_child(process):
    import signal
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def model_identity(path):
    path = Path(path).resolve()
    if not (path / 'config.json').is_file():
        raise ValueError('Incomplete model directory: ' + str(path))
    files = sorted(path.glob('*.safetensors'))
    if not files:
        raise ValueError('No local safetensors model: ' + str(path))
    return {'path': str(path), 'config_sha256': sha256(path / 'config.json'),
            'weights': {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in files},
            'tokenizer_sha256': sha256(path / 'tokenizer.json')}


def output_inventory(folder):
    result = {}
    for path in sorted(Path(folder).rglob('*')):
        relative = path.relative_to(folder)
        if (not path.is_file() or any(p == 'db' or p.startswith('db-resume-') or p in ('tool_logs', 'resources', '__pycache__') for p in relative.parts)
                or path.suffix in ('.lock', '.tmp') or path.name in ('CURRENT.json',)):
            continue
        if path.is_symlink():
            raise ValueError('Unexpected stage output symlink: ' + str(path))
        result[relative.as_posix()] = sha256(path)
    if not result:
        raise ValueError('Stage produced no verifiable outputs')
    return result


class Stages:
    def __init__(self, config, root, output, resume=False):
        import fcntl
        self.config, self.root, self.output, self.resume = config, Path(root).resolve(), Path(output).resolve(), resume
        if self.root not in self.output.parents or self.output.is_symlink():
            raise ValueError('Run must be a real directory inside this project')
        if self.output.exists() and any(self.output.iterdir()) and not resume:
            raise FileExistsError('Existing run requires --resume')
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / 'RUN.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.budget = DiskBudget(config['storage'])
        self.contract = {'config': config, 'code_sha256': manifest(config)['code_sha256'],
                         'models': {k: model_identity(v) for k, v in config['models'].items() if k != 'embedding'}}
        # Embeddings use a nested sentence-transformers layout, so their directory
        # metadata is recorded separately from the causal-model index format.
        self.contract['embedding'] = {str(p.relative_to(config['models']['embedding'])): [p.stat().st_size, p.stat().st_mtime_ns]
                for p in Path(config['models']['embedding']).rglob('*') if p.is_file() and p.suffix in ('.safetensors', '.bin', '.pt', '.json')}
        if not self.contract['embedding']:
            raise ValueError('Missing local embedding model')
        path = self.output / 'RUN_CONTRACT.json'
        if path.exists() and read_json(path) != self.contract:
            raise ValueError('Run configuration, source, or input model changed')
        if not path.exists():
            write_json(path, self.contract)

    def check_budget(self, reason):
        used = self.budget.check(0, reason)
        import importlib.util
        spec = importlib.util.spec_from_file_location('_memt_shared_cache_audit', self.root / 'scripts/audit_external_cache.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.shared_cache = module.audit(self.output, self.config)
        if used + self.shared_cache['observed_bytes'] > self.budget.limit:
            raise ValueError('Project plus conservatively counted shared caches exceeds budget')
        return used

    def state(self, stage, **fields):
        write_json(self.output / 'CURRENT.json', dict(stage=stage, time_unix=time.time(), pid=os.getpid(),
                   job=os.getenv('SLURM_JOB_ID'), **fields))

    def run(self, name, command, folder, inputs=(), env=None, resumable=False, minimum_seconds=0):
        folder = Path(folder)
        if self.output not in folder.resolve().parents:
            raise ValueError('Stage output outside run')
        command = list(map(str, command))
        receipt = self.output / 'stages' / (name + '.json')
        contract = {'command': command, 'inputs': {str(p): sha256(Path(p)) for p in inputs}}
        if receipt.exists():
            previous = read_json(receipt)
            if previous['contract'] != contract:
                raise ValueError('Stage inputs/command changed: ' + name)
            if previous.get('status') == 'complete':
                for relative, expected in previous['outputs_sha256'].items():
                    if sha256(folder / relative) != expected:
                        raise ValueError('Completed stage output changed: ' + name + '/' + relative)
                self.state(name, reused=True)
                return
        elif folder.exists() and any(folder.iterdir()):
            raise FileExistsError('Unrecorded nonempty stage: ' + str(folder))
        margin = self.config.get('study', {}).get('walltime_margin_seconds', 0)
        if margin and remaining_seconds() < max(margin, minimum_seconds + margin):
            self.state(name, deferred=True, reason='insufficient remaining allocation time')
            raise StageDeferred(name + ': insufficient remaining allocation time')
        attempt = (previous.get('attempt', 0) if receipt.exists() else 0) + 1
        if name == 'sft' and attempt > self.config.get('study', {}).get('maximum_sft_attempts', float('inf')):
            raise ValueError('Export-only SFT attempt limit reached; inspect interruption before restarting')
        if receipt.exists() and folder.exists() and any(folder.iterdir()) and not resumable:
            archived = self.output / 'interrupted' / (name + '-' + str(time.time_ns()))
            archived.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(archived)
            write_json(archived / 'STAGE_RECEIPT.json', read_json(receipt))
        self.check_budget('before stage ' + name)
        write_json(receipt, {'status': 'running', 'contract': contract, 'attempt': attempt, 'started_unix': time.time()})
        self.state(name, command=command)
        try:
            with (self.output / (name + '.log')).open('a', encoding='utf-8') as stream:
                process = subprocess.Popen(command, cwd=self.root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                while True:
                    try:
                        code = process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        try:
                            if margin and remaining_seconds() < margin:
                                raise StageDeferred(name + ': planned allocation boundary')
                            self.check_budget('running stage ' + name)
                            self.state(name, child_pid=process.pid)
                        except BaseException:
                            stop_child(process)
                            raise
            if code:
                raise subprocess.CalledProcessError(code, command)
            inventory = output_inventory(folder)
            write_json(receipt, {'status': 'complete', 'contract': contract, 'attempt': attempt, 'outputs_sha256': inventory,
                       'finished_unix': time.time()})
        except StageDeferred as error:
            write_json(receipt, {'status': 'deferred', 'contract': contract, 'attempt': attempt, 'reason': str(error), 'time_unix': time.time()})
            self.state(name, deferred=True, reason=str(error))
            raise
        except BaseException as error:
            write_json(receipt, {'status': 'failed', 'contract': contract, 'attempt': attempt, 'error': repr(error), 'time_unix': time.time()})
            self.state(name, failed=True, error=repr(error))
            raise

    def cli(self, command, config, folder, *args, **kwargs):
        import copy
        import sys
        name = kwargs.pop('name', command)
        path = self.output / 'resolved-configs' / (name + '.json')
        config = copy.deepcopy(config)
        config['storage']['ledger'] = str(self.output / 'ledgers' / (name + '.json'))
        write_json(path, config)
        return [sys.executable, '-u', '-m', 'cli', command, '--config', str(path), '--output', str(folder), *map(str, args)]
