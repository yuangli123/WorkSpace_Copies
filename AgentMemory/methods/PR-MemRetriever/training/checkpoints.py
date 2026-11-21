"""One recoverable checkpoint, one validation-selected model, and audited rotation."""
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from experiments.io import get_budget, read_json, sha256, write_json


def required_shards(state, optimizer=True):
    kinds = ('model', 'optim', 'extra_state') if optimizer else ('model',)
    return [f'actor/{kind}_world_size_{state["world_size"]}_rank_{rank}.pt'
            for kind in kinds for rank in range(state['world_size'])]


def seal_checkpoint(path, optimizer=True):
    path = Path(path)
    state = read_json(path / 'state.json')
    if not (path / 'COMPLETE').is_file():
        raise ValueError('Checkpoint save did not finish')
    for name in required_shards(state, optimizer):
        if not zipfile.is_zipfile(path / name):
            raise ValueError(f'Missing or truncated PyTorch archive: {name}')
    files = {}
    for file in sorted(path.rglob('*')):
        if file.is_symlink():
            raise ValueError(f'Checkpoint symlink: {file}')
        if file.is_file() and file.name not in ('FILE_MANIFEST.json', 'VALIDATED'):
            files[file.relative_to(path).as_posix()] = {'bytes': file.stat().st_size, 'sha256': sha256(file)}
    write_json(path / 'FILE_MANIFEST.json', {'step': state['step'], 'files': files,
                                           'model_only': not optimizer})
    write_json(path / 'VALIDATED', {'step': state['step'], 'manifest_sha256': sha256(path / 'FILE_MANIFEST.json')})
    return state


def verify_checkpoint(path, optimizer=True):
    path = Path(path)
    state, inventory = read_json(path / 'state.json'), read_json(path / 'FILE_MANIFEST.json')
    marker = read_json(path / 'VALIDATED')
    if not (path / 'COMPLETE').is_file() or marker['manifest_sha256'] != sha256(path / 'FILE_MANIFEST.json'):
        raise ValueError('Checkpoint is not validated and complete')
    if optimizer and (state.get('model_only') or inventory['model_only']):
        raise ValueError('Model-only selection cannot resume training')
    if inventory['step'] != state['step']:
        raise ValueError('Checkpoint step metadata mismatch')
    for name in required_shards(state, optimizer):
        if name not in inventory['files']:
            raise ValueError(f'Missing checkpoint inventory entry: {name}')
    for name, expected in inventory['files'].items():
        file = path / name
        if path.resolve() not in file.resolve().parents or file.is_symlink():
            raise ValueError(f'Unsafe checkpoint inventory path: {name}')
        if not file.is_file() or file.stat().st_size != expected['bytes'] or sha256(file) != expected['sha256']:
            raise ValueError(f'Checkpoint integrity failure: {name}')
    return state


class CheckpointManager:
    def __init__(self, output, updater, config):
        self.output = Path(output).resolve()
        self.updater, self.config = updater, config
        self.budget = get_budget()

    def event(self, kind, **values):
        write_json(self.output / 'events' / f'{time.time_ns()}-{uuid.uuid4().hex[:8]}.json',
                   dict(kind=kind, time_unix=time.time(), **values))

    def reserve(self, kind):
        if self.budget:
            self.budget.check(self.config['storage'][kind + '_reserve_bytes'], kind)

    def retire(self, path, reason):
        path = Path(path)
        if path.is_symlink() or self.output not in path.resolve().parents:
            raise ValueError('Refusing checkpoint retirement outside this run')
        files = []
        for file in path.rglob('*.pt'):
            if re.fullmatch(r'(model|optim)_world_size_\d+_rank_\d+\.pt', file.name):
                if file.is_symlink() or (hasattr(os, 'getuid') and file.stat().st_uid != os.getuid()):
                    raise ValueError(f'Unowned checkpoint weight: {file}')
                files.append(file)
        self.event('retire_checkpoint', path=str(path), reason=reason,
                   files=[{'path': str(file), 'bytes': file.stat().st_size} for file in files])
        complete = path / 'COMPLETE'
        if complete.exists():
            complete.replace(path / 'RETIRED_COMPLETE')
        for file in files:
            file.unlink()

    def save(self, state):
        path = self.output / 'checkpoints' / f'step-{state["step"]:06d}'
        if (path / 'COMPLETE').exists():
            seal_checkpoint(path)
        else:
            if path.exists():
                self.retire(path, 'discard incomplete save before retry')
            self.reserve('full_checkpoint')
            self.updater.save(path, state)
            seal_checkpoint(path)
        write_json(self.output / 'LATEST.json', {'path': str(path), 'step': state['step']})
        self.event('checkpoint_validated', path=str(path), step=state['step'], cursor=state['question_cursor'])
        for older in (self.output / 'checkpoints').glob('step-*'):
            if older != path and (older / 'COMPLETE').exists():
                self.retire(older, 'new complete checkpoint validated')
        return path

    def select(self, checkpoint, metrics):
        best_file = self.output / 'BEST.json'
        previous = read_json(best_file) if best_file.exists() else None
        if previous and metrics['f1'] <= previous['f1']:
            return previous
        state = read_json(Path(checkpoint) / 'state.json')
        target = self.output / 'best' / f'step-{state["step"]:06d}'
        if (target / 'COMPLETE').exists():
            verify_checkpoint(target, optimizer=False)
        else:
            if target.exists():
                self.retire(target, 'discard incomplete model-only copy')
            self.reserve('model')
            (target / 'actor').mkdir(parents=True, exist_ok=True)
            for name in required_shards(state, optimizer=False):
                # Saved shards are immutable; rotation unlinks only the checkpoint name.
                os.link(Path(checkpoint) / name, target / name)
            shutil.copytree(Path(checkpoint) / 'actor/huggingface', target / 'actor/huggingface', dirs_exist_ok=True)
            write_json(target / 'state.json', dict(state, model_only=True, selection='validation_f1', validation=metrics))
            write_json(target / 'COMPLETE', {'source_checkpoint': str(checkpoint)})
            seal_checkpoint(target, optimizer=False)
        selected = dict(path=str(target), step=state['step'], f1=metrics['f1'])
        write_json(best_file, selected)
        self.event('best_model_selected', **selected)
        if previous and previous['path'] != str(target):
            self.retire(previous['path'], 'new validation best model validated')
        return selected
