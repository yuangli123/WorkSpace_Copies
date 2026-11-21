"""Stop at an update boundary and verify an actual distributed restore round trip."""
import contextlib
import gzip
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from training.checkpoints import required_shards, seal_checkpoint
from experiments.io import get_budget, write_json


class ResourceMonitor:
    def __init__(self, output, config):
        self.output, self.config = Path(output), config
        self.interval = config['runtime'].get('monitor_interval', 2)
        self.stopped = threading.Event()
        self.peaks = {}
        self.error = None

    def __enter__(self):
        self.output.joinpath('resources').mkdir(parents=True, exist_ok=True)
        write_json(self.output / 'RUNNING.json', dict(host=os.uname().nodename if hasattr(os, 'uname') else 'local',
                                                      job=os.getenv('SLURM_JOB_ID'), **resource_snapshot()))
        if self.interval:
            self.thread = threading.Thread(target=self.sample, daemon=True)
            self.thread.start()
        return self

    def sample(self):
        budget = get_budget()
        written = 1024 * 1024
        while not self.stopped.is_set():
            value = resource_snapshot()
            try:
                if budget and written >= 1024 * 1024:
                    budget.check(2 * 1024 * 1024, 'resource telemetry')
                    written = 0
                command = ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                           '--format=csv,noheader,nounits']
                if os.getenv('CUDA_VISIBLE_DEVICES'):
                    command.extend(['--id', os.environ['CUDA_VISIBLE_DEVICES']])
                gpu = subprocess.run(command, text=True, capture_output=True, timeout=5)
                value['gpu'] = gpu.stdout
                value['gpu_error'] = gpu.stderr
                for row in gpu.stdout.splitlines():
                    cells = row.split(',')
                    if len(cells) == 4 and cells[1].strip().isdigit():
                        key = 'gpu_' + cells[0].strip() + '_used_mib'
                        self.peaks[key] = max(self.peaks.get(key, 0), int(cells[1]))
                memory = value.get('cgroup_memory', {}).get('memory.usage_in_bytes', 0)
                self.peaks['cgroup_observed_bytes'] = max(memory, self.peaks.get('cgroup_observed_bytes', 0))
                text = json.dumps(value) + '\n'
                with gzip.open(self.output / 'resources/timeline.jsonl.gz', 'at', encoding='utf-8') as stream:
                    stream.write(text)
                written += len(text.encode('utf-8'))
            except (OSError, subprocess.TimeoutExpired) as error:
                self.error = str(error)
            except Exception as error:
                self.error = str(error)
                print('RESOURCE_MONITOR_STOPPED ' + self.error, flush=True)
                return
            self.stopped.wait(self.interval)

    def __exit__(self, *args):
        self.stopped.set()
        if self.interval:
            self.thread.join(timeout=10)
        write_json(self.output / 'resources/peaks.json', dict(peaks=self.peaks, error=self.error))


@contextlib.contextmanager
def stop_requests(config):
    requested, handlers = [False], {}
    def request(signum, frame):
        if requested[0] and signum == getattr(signal, 'SIGINT', None):
            raise KeyboardInterrupt('Repeated interrupt; keep last complete checkpoint')
        requested[0] = True
        print('STOP_REQUESTED: finishing the current update before checkpointing', flush=True)
    for name in ('SIGINT', 'SIGTERM', 'SIGUSR1'):
        signum = getattr(signal, name, None)
        if signum is not None:
            handlers[signum] = signal.signal(signum, request)
    try:
        yield lambda: requested[0] or Path(config['runtime']['stop_file']).exists()
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)


def resource_snapshot():
    result = {'time_unix': time.time(), 'pid': os.getpid()}
    groups = Path('/proc/self/cgroup')
    if groups.exists():
        for line in groups.read_text().splitlines():
            _, controllers, group = line.split(':', 2)
            if controllers == 'memory':
                folder = Path('/sys/fs/cgroup/memory') / group.lstrip('/')
                result['cgroup_memory'] = {name: int((folder / name).read_text()) for name in
                                          ('memory.usage_in_bytes', 'memory.max_usage_in_bytes',
                                           'memory.limit_in_bytes', 'memory.failcnt') if (folder / name).exists()}
    return result


def compare_state(left, right, location='root'):
    import numpy as np
    import torch
    if hasattr(left, '_local_tensor'):
        assert left.shape == right.shape and left.placements == right.placements, location
        return compare_state(left._local_tensor, right._local_tensor, location)
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f'Restored tensor mismatch: {location}')
        return left.numel()
    if isinstance(left, np.ndarray):
        assert np.array_equal(left, right), location
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), location
        return sum(compare_state(left[key], right[key], f'{location}/{key}') for key in left)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), location
        return sum(compare_state(a, b, f'{location}/{i}') for i, (a, b) in enumerate(zip(left, right)))
    else:
        assert left == right, location
    return 0


def verify_live_restore(manager, original, state):
    import gc
    import torch
    probe = manager.output / 'checkpoints' / f'resume-probe-{state["step"]:06d}'
    if probe.exists():
        manager.retire(probe, 'discard previous restore probe')
    manager.reserve('full_checkpoint')
    manager.updater.save(probe, state)
    seal_checkpoint(probe)
    tensors = {}
    for name in required_shards(state):
        left = torch.load(Path(original) / name, map_location='cpu', mmap=True, weights_only=False)
        right = torch.load(probe / name, map_location='cpu', mmap=True, weights_only=False)
        tensors[name] = compare_state(left, right, name)
        del left, right
        gc.collect()
    report = dict(step=state['step'], question_cursor=state['question_cursor'], order=state['order'],
                  original=str(original), probe=str(probe), tensors_compared=tensors,
                  model_optimizer_scheduler_rng_equal=True, **resource_snapshot())
    write_json(manager.output / 'RESTORE_VERIFIED.json', report)
    manager.event('restore_roundtrip_verified', step=state['step'], cursor=state['question_cursor'])
    manager.retire(probe, 'full restore comparison passed; keep original latest checkpoint')
    manager.updater.last_checkpoint = Path(original)
