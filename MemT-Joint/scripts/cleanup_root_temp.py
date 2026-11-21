"""Relocate only owned, inactive numeric t/r job directories into project artifacts."""
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

root = Path('/public/home/wwxia/yuang.li').resolve()
assert Path.cwd().resolve() == root / 'workspace/MemT-Joint'
assert root.stat().st_uid == os.getuid()
store = root / 'workspace/MemT-Joint/artifacts/tmp'
assert store.resolve() == store and root in store.parents
store.mkdir(parents=True, exist_ok=True)
assert store.stat().st_uid == os.getuid() and store.stat().st_dev == root.stat().st_dev
report = Path(sys.argv[1]).resolve()
assert root in report.parents and not report.exists()
candidates = [p for p in root.iterdir() if re.fullmatch(r'[tr][0-9]+', p.name)]
jobs = sorted({p.name[1:] for p in candidates})
states = {}
if jobs:
    result = subprocess.run(['sacct', '-j', ','.join(jobs), '-n', '-P', '--format=JobIDRaw,State'],
                            text=True, capture_output=True, check=True)
    for line in result.stdout.splitlines():
        job, state = line.split('|')[:2]
        if job in jobs:
            states[job] = state.split()[0].rstrip('+')
active = subprocess.run(['squeue', '-u', str(os.getuid()), '-h', '-o', '%i'],
                        text=True, capture_output=True, check=True).stdout.split()
live = {p.name: set() for p in candidates}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        references = [(proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')]
        references.append(os.readlink(proc / 'cwd'))
        for fd in (proc / 'fd').iterdir():
            try:
                references.append(os.readlink(fd))
            except OSError:
                pass
        for p in candidates:
            if any(str(p) + '/' in value or value == str(p) for value in references):
                live[p.name].add(int(proc.name))
    except (OSError, ProcessLookupError):
        pass
terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED', 'DEADLINE'}
archive = store / ('retired-root-' + time.strftime('%Y%m%dT%H%M%S') + '-' + str(os.getpid()))
archive.mkdir()
records = []
for p in sorted(candidates):
    assert p.parent == root and not p.is_symlink() and p.is_dir() and not p.is_mount()
    assert p.stat().st_uid == os.getuid()
    row = {'name': p.name, 'job_state': states.get(p.name[1:]), 'live_reference_pids': sorted(live[p.name]),
           'files': 0, 'bytes': 0, 'action': 'pending'}
    if p.name[1:] in active or row['job_state'] not in terminal or row['live_reference_pids']:
        row['action'] = 'kept_active_or_unverified'
    else:
        for directory, dirs, files in os.walk(p, followlinks=False):
            for name in dirs + files:
                child = Path(directory) / name
                value = child.lstat()
                assert value.st_uid == os.getuid(), str(child)
                assert child.is_symlink() or not child.is_mount(), str(child)
                row['files'] += not stat.S_ISDIR(value.st_mode)
                row['bytes'] += value.st_size if stat.S_ISREG(value.st_mode) else 0
        destination = archive / p.name
        assert destination.resolve().parent == archive and not destination.exists()
        row['destination'] = str(destination)
        row['action'] = 'verified_for_relocation'
    records.append(row)

def save():
    report.write_text(json.dumps({'archive': str(archive), 'directories': records,
        'remaining_root_t_r': sorted(p.name for p in root.iterdir() if re.fullmatch(r'[tr][0-9]+', p.name))}, indent=2))

save()
for row in records:
    if row['action'] == 'verified_for_relocation':
        source = root / row['name']
        assert source.resolve().parent == root and source.stat().st_uid == os.getuid()
        os.rename(source, Path(row['destination']))
        row['action'] = 'relocated'
        save()
alias = root / '.m'
if os.path.lexists(alias):
    assert alias.is_symlink() and alias.lstat().st_uid == os.getuid() and alias.resolve() == store
else:
    alias.symlink_to(store, target_is_directory=True)
assert len(str(alias / 'r999999/ray/session_2026-11-07_23-59-59_999999_999999/sockets/plasma_store').encode()) < 108
print('ROOT_TEMP_RELOCATED', sum(x['action'] == 'relocated' for x in records), str(archive), flush=True)
print('TEMP_STORAGE', str(store), 'SHORT_ALIAS', str(alias), flush=True)
