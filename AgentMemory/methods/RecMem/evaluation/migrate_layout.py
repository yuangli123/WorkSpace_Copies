"""Preview or migrate legacy AgentMemory experiment artifacts; never delete the source tree."""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from evaluation.protocol import atomic_json


ANSWER_KEYS = {
    "method", "bench", "model", "judge_model", "conv_limit", "sample_size",
    "dataset_path", "dataset_sha256", "llm_base_url", "embedding_model",
    "embedding_base_url", "source_version", "context_budget", "top_k",
    "rag_store", "embedding_dim", "semantic_store", "subconscious_store",
    "episodic_store", "SEMANTIC_STORE", "SUBCONSCIOUS_STORE", "EPISODIC_STORE",
    "min_consolidation_cnt", "min_relevant_score", "retrieve_raw_topk",
    "retrieve_epi_topk", "merge_with_epi_thresh", "disable_semantic_refinement",
    "semantic_memory_topk", "semantic_memory_threshold", "search_only",
    "store_root", "question_ids", "adaptation",
}
SCORE_KEYS = {"bench", "judge_model", "dataset_sha256", "answer_file", "answer_sha256"}
PATH_KEYS = {"dataset_path", "rag_store", "semantic_store", "subconscious_store",
             "episodic_store", "SEMANTIC_STORE", "SUBCONSCIOUS_STORE", "EPISODIC_STORE", "store_root"}
FALLBACK_STORES = {"SEMANTIC_STORE": "semantic", "SUBCONSCIOUS_STORE": "subconscious",
                   "EPISODIC_STORE": "episodic"}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target(path, old, root):
    """Map only known legacy artifact locations; refuse unknown absolute paths."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"Expected an absolute path: {path}")
    try:
        parts = path.relative_to(old).parts
    except ValueError as error:
        raise ValueError(f"Path is outside the legacy project: {path}") from error
    if not parts:
        raise ValueError(f"Cannot migrate project root: {path}")
    if parts[0] == "dataset":
        return root / "artifacts/data" / Path(*parts[1:])
    if parts[0] == ".cache":
        return root / "artifacts/cache" / Path(*parts[1:])
    if parts[0] == "logs":
        return root / "artifacts/logs" / Path(*parts[1:])
    if parts[0] == ".env" and len(parts) == 1:
        return root / ".env"
    if parts[0] == "results":
        if "stores" in parts[1:]:
            index = parts.index("stores")
            return root / "artifacts/stores" / Path(*parts[1:index], *parts[index + 1:])
        if len(parts) > 1 and parts[1] == "service-logs":
            return root / "artifacts/logs/service-logs" / Path(*parts[2:])
        return root / "artifacts/results" / Path(*parts[1:])
    raise ValueError(f"Unmapped legacy artifact path: {path}")


def manifest_path(key, value, old, root):
    if key in FALLBACK_STORES:
        layer = FALLBACK_STORES[key]
        stale = root.parent / "RecMem/results/qdrant" / layer
        if Path(value) == stale:
            return str(root / "artifacts/stores/recmem" / layer)
    return str(target(value, old, root))


def current_versions(root):
    from evaluation.external_runner import source_version as external_version
    from evaluation.run_experiments import source_version as native_version
    versions = {name: native_version() for name in ("recmem", "fullcontext", "naive_rag")}
    versions.update({name: external_version(name, root / "methods")
                     for name in ("memoryos", "mem0", "amem")})
    return versions


def inspect(old_project, root, versions=None):
    old, root = Path(old_project).resolve(), Path(root).resolve()
    if old != root / "RecMem" or not (root / "methods/RecMem").is_dir():
        raise ValueError("Expected legacy AgentMemory/RecMem and deployed AgentMemory/methods/RecMem")
    personal = os.getenv("PERSONAL_ROOT")
    if personal and not root.is_relative_to((Path(personal) / "workspace").resolve()):
        raise ValueError("Migration must stay in PERSONAL_ROOT/workspace")
    versions = versions or current_versions(root)
    sources = []
    for name in ("dataset", ".cache", "logs", "results", ".env"):
        source = old / name
        if source.exists():
            sources.append(source)
    if not sources:
        raise ValueError("No legacy artifacts found")
    files = []
    for source in sources:
        entries = [source] if source.is_file() else list(source.rglob("*"))
        if source.is_symlink() or any(path.is_symlink() for path in entries):
            raise ValueError(f"Refusing symlink in {source}")
        files.extend(path for path in entries if path.is_file())
    if not files:
        raise ValueError("No legacy artifacts found")
    destinations = set()
    for source in files:
        if source.is_symlink():
            raise ValueError(f"Refusing symlink: {source}")
        destination = target(source, old, root)
        if destination in destinations:
            raise ValueError(f"Two legacy files map to {destination}")
        destinations.add(destination)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Target already exists: {destination}")
        parent = destination.parent
        while parent != root:
            if parent.exists() and not parent.is_dir():
                raise FileExistsError(f"Target parent is not a directory: {parent}")
            parent = parent.parent
    manifests = {}
    for source in (old / "results").rglob("*.run.json") if (old / "results").is_dir() else ():
        info = json.loads(source.read_text(encoding="utf-8"))
        scored = "answer_file" in info
        allowed = SCORE_KEYS if scored else ANSWER_KEYS
        if set(info) - allowed:
            raise ValueError(f"Unknown manifest fields in {source}: {sorted(set(info) - allowed)}")
        if scored:
            answer = Path(info["answer_file"])
            if not answer.is_file() or sha(answer) != info["answer_sha256"]:
                raise ValueError(f"Score answer hash mismatch: {source}")
            answer_manifest = json.loads(Path(str(answer) + ".run.json").read_text(encoding="utf-8"))
            if info["dataset_sha256"] != answer_manifest["dataset_sha256"]:
                raise ValueError(f"Score dataset hash mismatch: {source}")
            updated = {**info, "answer_file": str(target(answer, old, root))}
        else:
            answer = Path(str(source)[:-9])
            dataset = Path(info["dataset_path"])
            if not answer.is_file() or not dataset.is_file() or sha(dataset) != info["dataset_sha256"]:
                raise ValueError(f"Answer or dataset missing/hash mismatch: {source}")
            method = info["method"]
            if method not in versions or not info.get("source_version"):
                raise ValueError(f"Unknown method/source version: {source}")
            updated = {key: manifest_path(key, value, old, root) if key in PATH_KEYS and value else value
                       for key, value in info.items()}
            record = target(answer, old, root).with_name(target(answer, old, root).name + ".migration.json")
            if record.exists():
                raise FileExistsError(record)
        destination = target(source, old, root)
        backup = root / "artifacts/migration/backup" / source.relative_to(old)
        if backup.exists():
            raise FileExistsError(backup)
        manifests[source] = (updated, backup)
    return old, root, files, manifests, versions


def migrate(old_project, root, apply=False, versions=None):
    old, root, files, manifests, versions = inspect(old_project, root, versions)
    summary = {"files": len(files), "manifests": len(manifests), "old_project": str(old),
               "new_root": str(root), "applied": apply}
    if not apply:
        return summary
    # Recheck all targets before the first write; do not overwrite any pre-existing file.
    for source in files:
        if target(source, old, root).exists():
            raise FileExistsError(target(source, old, root))
    for source, (_, backup) in manifests.items():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
    critical = {source: sha(source) for source in files
                if (old / "dataset") in source.parents or Path(str(source) + ".run.json") in manifests}
    # Move files, not parent directories: the old project remains available for audit.
    for source in files:
        destination = target(source, old, root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    for source, digest in critical.items():
        if sha(target(source, old, root)) != digest:
            raise ValueError(f"Migrated file hash mismatch: {source}")
    for source, (updated, backup) in manifests.items():
        destination = target(source, old, root)
        original = json.loads(backup.read_text(encoding="utf-8"))
        atomic_json(str(destination), updated)
        if "answer_file" not in original:
            old_answer = Path(str(source)[:-9])
            new_answer = target(old_answer, old, root)
            completed = sorted(json.loads(new_answer.read_text(encoding="utf-8")))
            atomic_json(str(new_answer) + ".migration.json", {
                "old_answer_file": str(old_answer), "answer_file": str(new_answer),
                "backup_manifest": str(backup), "original_manifest_sha256": sha(backup),
                "migrated_manifest_sha256": sha(destination),
                "old_source_version": original["source_version"],
                "new_source_version": versions[original["method"]],
                "dataset_sha256": original["dataset_sha256"],
                "completed_before_migration": completed,
                "migrated_at": datetime.now(timezone.utc).isoformat(),
            })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-project", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--apply", action="store_true", help="Perform the checked migration (default: preview only)")
    args = parser.parse_args(argv)
    print(json.dumps(migrate(args.old_project, args.new_root, args.apply), indent=2))


if __name__ == "__main__":
    main()
