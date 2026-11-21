"""Conservative accounting for the explicitly charged experiment directories."""
import json
import os
import shutil
import stat
import threading
import time
from pathlib import Path


class StorageBudgetExceeded(RuntimeError):
    pass


def tree_bytes(path, seen=None, links=None):
    seen = set() if seen is None else seen
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    pending = [path]
    while pending:
        item = pending.pop()
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue  # Ray temporary files and retired checkpoint names can vanish during a scan.
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        total += max(info.st_size, getattr(info, 'st_blocks', 0) * 512)
        if stat.S_ISLNK(info.st_mode) and links is not None:
            links.append((item, item.resolve()))
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(item) as entries:
                    pending.extend(Path(entry.path) for entry in entries)
            except FileNotFoundError:
                continue
    return total


class DiskBudget:
    def __init__(self, config):
        self.allowed = Path(config['allowed_root']).resolve()
        self.lock = threading.RLock()
        self.roots = [Path(path).resolve() for path in config['roots']]
        for path in self.roots:
            if path != self.allowed and self.allowed not in path.parents:
                raise ValueError(f'Budget root outside personal workspace: {path}')
        self.limit = config['limit_bytes']
        self.excluded = [Path(path).resolve() for path in config.get('excluded_roots', [])]
        self.warning = config['warning_bytes']
        self.reserve = config.get('runtime_reserve_bytes', 2_000_000_000)
        self.ledger = Path(config['ledger']).resolve()
        self.check_path(self.ledger)
        self.history = json.loads(self.ledger.read_text()) if self.ledger.exists() else {}
        self.history.update(limit_bytes=self.limit, warning_bytes=self.warning,
                            roots=list(map(str, self.roots)), runtime_reserve_bytes=self.reserve,
                            excluded_roots=list(map(str, self.excluded)),
                            accounting='max(apparent,allocated) bytes, inode deduplicated, symlink targets not traversed')
        self.check(0, 'initial inventory')

    def check_path(self, path):
        path = Path(path).resolve()
        if not any(path == root or root in path.parents for root in self.roots):
            raise ValueError(f'Write outside charged experiment roots: {path}')

    def check(self, additional=0, reason='write'):
        with self.lock:
            return self._check(additional, reason)

    def _check(self, additional, reason):
        seen, details, links = set(), {}, []
        for root in self.roots:
            details[str(root)] = tree_bytes(root, seen, links)
        for link, target in links:
            if target.exists() and not any(target == root or root in target.parents for root in self.roots + self.excluded):
                raise ValueError(f'Unaccounted symlink target: {link} -> {target}')
        used = sum(details.values())
        expected = used + additional + self.reserve
        self.history.update(time_unix=time.time(), observed_bytes=used, expected_bytes=expected,
                            additional_bytes=additional, reason=reason, by_root=details,
                            peak_observed_bytes=max(used, self.history.get('peak_observed_bytes', 0)),
                            peak_requested_bytes=max(expected, self.history.get('peak_requested_bytes', 0)),
                            peak_admitted_bytes=max(expected if expected <= self.limit else 0,
                                                    self.history.get('peak_admitted_bytes', 0)))
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.ledger.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.history, indent=2), encoding='utf-8')
        temporary.replace(self.ledger)
        if expected > self.limit:
            raise StorageBudgetExceeded(f'{reason}: observed={used}, planned={additional}, reserve={self.reserve}, limit={self.limit}')
        if max(used, expected) >= self.warning:
            print(f'STORAGE_WARNING observed={used} reserved_peak={expected} limit={self.limit}', flush=True)
        existing = next((root for root in self.roots if root.exists()), self.allowed)
        if shutil.disk_usage(existing).free < additional + self.reserve:
            raise StorageBudgetExceeded(f'{reason}: filesystem free space is insufficient')
        return used
