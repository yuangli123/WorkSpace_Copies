import hashlib
import gzip
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


_budget = None


def configure_storage(config):
    global _budget
    from experiments.storage import DiskBudget
    # Only rank zero writes shared manifests and admits collective trainer saves.
    _budget = DiskBudget(config['storage']) if config.get('storage') and int(os.getenv('RANK', '0')) == 0 else None
    return _budget


def get_budget():
    return _budget


def reserve_write(path, count, reason):
    if _budget:
        _budget.check_path(path)
        _budget.check(count, reason)


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2)
    reserve_write(path, len(text.encode('utf-8')) + 65536, 'JSON ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if str(path).endswith('.gz'):
        with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
            stream.write(text)
    else:
        temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    return list(iter_jsonl(path))


def iter_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    reserve_write(path, 5 * 1024 * 1024, 'JSONL ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if str(path).endswith('.gz') else open
    encoder = json.JSONEncoder(ensure_ascii=False)
    with opener(temporary, 'wt', encoding='utf-8') as stream:
        buffer, size, since_check = [], 0, 0
        for row in rows:
            for chunk in encoder.iterencode(row):
                buffer.append(chunk)
                size += len(chunk)
                if size >= 65536:
                    text = ''.join(buffer)
                    count = len(text.encode('utf-8'))
                    if count > 4 * 1024 * 1024:
                        reserve_write(path, count + count // 100 + 65536, 'large JSONL chunk ' + str(path))
                    stream.write(text)
                    since_check += count
                    buffer, size = [], 0
                    if since_check >= 4 * 1024 * 1024:
                        stream.flush()
                        reserve_write(path, 5 * 1024 * 1024, 'JSONL continuation ' + str(path))
                        since_check = 0
            buffer.append('\n')
            size += 1
        stream.write(''.join(buffer))
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def manifest(config, inputs=()):
    packages = {}
    for name in ("torch", "transformers", "accelerate", "vllm", "ray", "chromadb", "sentence-transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    code = {p.relative_to(root).as_posix(): sha256(p) for p in root.glob("*.py")}
    for package in ('models', 'memory', 'methods', 'training', 'evaluation', 'experiments'):
        code.update({p.relative_to(root).as_posix(): sha256(p) for p in (root / package).rglob('*.py')})
    code.update({p.relative_to(root).as_posix(): sha256(p) for p in (root / 'scripts').glob('*.py')})
    for relative in ("third_party/Tree-GRPO/verl/workers/fsdp_workers.py",
                     "third_party/Tree-GRPO/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
                     "third_party/Tree-GRPO/verl/single_controller/ray/base.py",
                     "third_party/Tree-GRPO/verl/workers/sharding_manager/fsdp_vllm.py",
                     "third_party/Tree-GRPO/verl/utils/checkpoint/fsdp_checkpoint_manager.py",
                     "third_party/Tree-GRPO/verl/utils/torch_functional.py",
                     "third_party/Tree-GRPO/scripts/merge_ckpt/model_merger.py",
                     "third_party/Tree-GRPO/verl/trainer/config/ppo_trainer.yaml"):
        code[relative] = sha256(root / relative)
    return {"config": config, "python": platform.python_version(), "packages": packages,
            "conda_prefix": os.environ.get("CONDA_PREFIX"), "code_sha256": code,
            "inputs": {str(Path(p).resolve()): sha256(p) for p in inputs},
            "upstream": read_json(root / "UPSTREAM.lock.json")}


def checked_output(path, resume=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path):
    from models.roles import load_config as load
    return load(path)
