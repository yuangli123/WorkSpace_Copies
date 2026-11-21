"""Input provenance and JSON output helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def visible_gpu_ids(minimum: int = 4) -> list[str]:
    values = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(values) < minimum:
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES must expose at least {_COUNT_WORDS.get(minimum, minimum)} GPUs "
            f"for the requested {minimum}-GPU condition"
        )
    if len(set(values)) != len(values):
        raise SystemExit("CUDA_VISIBLE_DEVICES contains duplicate GPU identifiers")
    return values


def model_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        if any(part.startswith('.') for part in relative.parts) or relative.name == 'DOWNLOAD_VERIFIED.json':
            continue
        if item.is_file():
            digest.update(relative.as_posix().encode())
            with item.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def review_core_hash(root: Path) -> str:
    """Fingerprint the frozen review implementation and its rule/config files."""
    digest = hashlib.sha256()
    for directory in (root / "review", root / "config"):
        if not directory.is_dir():
            continue
        for item in sorted(directory.rglob("*")):
            if item.is_file() and item.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".md", ".txt"}:
                digest.update(item.relative_to(root).as_posix().encode("utf-8"))
                with item.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
    return digest.hexdigest()
