"""Conservatively count entire shared CUDA/Torch caches without changing them."""
import argparse
import os
import stat
import sys
import tempfile
import time
from pathlib import Path


def snapshot(folders):
    def fail(error):
        raise error
    seen, totals = set(), {}
    for folder in folders:
        total = 0
        if folder.is_symlink():
            raise ValueError(f'Unexpected cache symlink: {folder}')
        if not folder.exists():
            totals[str(folder)] = 0
            continue
        if not folder.is_dir():
            raise ValueError(f'Cache root is not a directory: {folder}')
        for parent, directories, files in os.walk(folder, onerror=fail):
            for name in directories + files:
                file = Path(parent) / name
                try:
                    value = file.lstat()
                except FileNotFoundError:
                    continue  # Concurrent cache eviction removed the entry.
                if stat.S_ISLNK(value.st_mode):
                    raise ValueError(f'Unexpected cache symlink: {file}')
                identity = (value.st_dev, value.st_ino)
                if stat.S_ISREG(value.st_mode) and identity not in seen:
                    total += max(value.st_size, value.st_blocks * 512)
                    seen.add(identity)
        totals[str(folder)] = total
    return {'time_unix': time.time(), 'by_root': totals, 'observed_bytes': sum(totals.values())}


def audit(run, config):
    from experiments.io import read_json, write_json
    home = Path('/public/home/wwxia')
    value = snapshot([home / '.nv/ComputeCache', home / '.cache/torch'])
    target = run / 'preparation/shared-cache-audit.json'
    previous = read_json(target) if target.exists() else {}
    value['peak_observed_bytes'] = max(previous.get('peak_observed_bytes', 0), value['observed_bytes'])
    value['accounting'] = 'Entire shared caches counted conservatively; files are only stat-ed, never modified.'
    value['budget_cover'] = 'Included within the existing runtime reserve; add to project observed peaks for final audit.'
    write_json(target, value)
    assert value['peak_observed_bytes'] <= config['storage']['runtime_reserve_bytes'], 'Shared caches exceed reserved allowance'
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'one').write_bytes(b'x' * 4096)
            size = max((root / 'one').stat().st_size, (root / 'one').stat().st_blocks * 512)
            os.link(root / 'one', root / 'two')
            assert snapshot([root, root / 'absent'])['observed_bytes'] == size
            (root / 'link').symlink_to(root / 'one')
            try:
                snapshot([root])
            except ValueError:
                pass
            else:
                raise AssertionError('Cache symlink silently excluded')
        print('shared cache audit self-test passed')
        return
    root = Path(__file__).resolve().parents[1]
    assert Path.cwd().resolve() == root and os.getuid() == 2086
    assert args.run and args.config
    run = args.run.resolve()
    assert run.is_relative_to(root / 'artifacts')
    sys.path.insert(0, str(root))
    from experiments.io import configure_storage, load_config
    config = load_config(args.config)
    config['storage']['ledger'] = str(run / 'storage-cache-audit.json')
    configure_storage(config)
    print(audit(run, config))


if __name__ == '__main__':
    main()
