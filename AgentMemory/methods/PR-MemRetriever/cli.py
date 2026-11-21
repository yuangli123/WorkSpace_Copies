from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from experiments.io import checked_output, load_config, manifest, read_jsonl, write_json


def get_config(args):
    import yaml
    config = load_config(args.config)
    for override in args.set:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Overrides must use dotted.key=value")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    from models.roles import normalize
    config = normalize(config)
    if config["retrieval"]["evidence_alpha"] is None or config["retrieval"]["expansion_rounds"] is None:
        raise ValueError("Set the undocumented evidence_alpha and expansion_rounds explicitly; see docs/ASSUMPTIONS.md")
    if config["model"]["temperature"] <= 0:
        raise ValueError("Training sampling temperature must be positive; evaluation explicitly uses greedy decoding")
    for section, keys in {"retrieval": ["trees", "max_depth", "pivots", "steps", "batch_size", "save_every", "max_prompt_tokens", "max_observation_tokens"],
                          "construction": ["rollouts", "steps", "max_length", "per_device_batch_size", "gradient_accumulation", "save_every", "maximum_operations"],
                          "memory": ["summary_context_turns", "update_retrieval_topk", "retrieval_topk"],
                          "distributed": ["gpus", "tensor_parallel", "inference_tensor_parallel"]}.items():
        for key in keys:
            value = config[section][key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    cfg = config["retrieval"]
    if not isinstance(cfg["expansion_rounds"], int) or cfg["expansion_rounds"] < 0:
        raise ValueError("expansion_rounds must be a nonnegative integer")
    if not math.isfinite(cfg["evidence_alpha"]) or cfg["evidence_alpha"] < 0:
        raise ValueError("evidence_alpha must be finite and nonnegative")
    if not 0 < config["construction"]["validation_fraction"] < 1:
        raise ValueError("SFT validation_fraction must be in (0,1)")
    if config["distributed"]["gpus"] % config["distributed"]["tensor_parallel"]:
        raise ValueError("GPU count must be divisible by tensor parallel size")
    return config


def policy(args, config):
    if not Path(args.model).is_dir():
        raise ValueError("Use an existing local model directory; download weights into your own models/ first")
    if args.inference_engine == "vllm":
        from models.vllm import VLLMPolicy
        return VLLMPolicy(args.model, config)
    from models.hf import HFPolicy
    return HFPolicy(args.model, config)


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('run-construction', 'run-construction-study'):
        if argv[0] == 'run-construction-study':
            from experiments.construction_study import main as run
        else:
            from experiments.construction import main as run
        return run(argv[1:])
    parser = argparse.ArgumentParser(description="Mem-T training research reimplementation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser('run-construction', help='Run the versioned construction training and evaluation pipeline')
    sub.add_parser('run-construction-study', help='Run formal construction SFT, then C200+R0 and C200+R200 tests')
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--benchmark", choices=["locomo", "hotpotqa", "longmemeval", "narrativeqa"], default="locomo")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    for name in ("build-memory", "train-retrieval", "collect-trees", "derive-sft", "train-sft", "evaluate", "export-llamafactory", "doctor"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/memt4b.yaml")
        command.add_argument("--set", action="append", default=[])
        if name != "doctor":
            command.add_argument("--output", required=True)
        if name in {"build-memory", "train-retrieval", "collect-trees", "train-sft", "evaluate", "export-llamafactory"}:
            command.add_argument("--model")
            command.add_argument("--inference-engine", choices=["hf", "vllm"], default="hf")
        if name in {"train-retrieval", "collect-trees", "derive-sft", "evaluate"}:
            command.add_argument("--memory", required=True)
        if name == "build-memory":
            command.add_argument("--data", required=True)
        if name == "train-retrieval":
            command.add_argument("--backend", choices=["hf", "verl"], default="verl")
            command.add_argument("--valid-memory", required=True)
            command.add_argument("--resume")
        if name == "derive-sft":
            command.add_argument("--forests", required=True)
        if name in {"train-sft", "export-llamafactory"}:
            command.add_argument("--selected", required=True)
        if name == "train-sft":
            command.add_argument("--resume")
        if name == "evaluate":
            command.add_argument("--parallel-calls", type=int, choices=(1, 2, 4, 8), default=1,
                                 help="Maximum tool calls per retrieval decision")
            command.add_argument("--serial-tool-execution", action="store_true",
                                 help="Run each planned batch serially for timing comparison")
            command.add_argument("--limit", type=int, help="Evaluate only the first N fixed questions")
        if name in ('build-memory', 'collect-trees', 'evaluate'):
            command.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from memory.data import prepare_locomo, prepare_ood
        if args.benchmark == "locomo":
            prepare_locomo(args.source, args.output)
        else:
            prepare_ood(args.source, args.output, args.benchmark)
        return
    config = get_config(args)
    if hasattr(args, 'model') and args.model is None:
        if 'models' not in config:
            parser.error('--model is required for a legacy configuration')
        role = ('construction' if args.command in ('build-memory', 'train-sft', 'export-llamafactory')
                else 'base' if args.command == 'train-retrieval' else 'retrieval')
        args.model = config['models'][role]
    from experiments.io import configure_storage
    configure_storage(config)
    if hasattr(args, "model") and not Path(args.model).is_dir():
        raise ValueError("--model must be an existing local model directory")
    from experiments.workflow import MemoryRun, build, collect_forests, derive_sft, evaluate, seed_all, train_retrieval
    seed_all(config["seed"])
    if args.command == "doctor":
        import importlib.util
        import sys
        import torch
        print(json.dumps({"python": sys.version, "executable": sys.executable, "conda_prefix": os.environ.get("CONDA_PREFIX"),
                          "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count(),
                          "model_configuration": config["model"],
                          "modules": {name: importlib.util.find_spec(name) is not None for name in ("transformers", "vllm", "ray", "chromadb")}}, indent=2))
    elif args.command == "build-memory":
        config["model"]["checkpoint"] = str(Path(args.model).resolve())
        build(config, args.data, args.output, policy(args, config), resume=args.resume)
    elif args.command == "train-retrieval":
        memory, valid = MemoryRun(args.memory, config), MemoryRun(args.valid_memory, config)
        if valid.partition["split"] != "valid":
            raise ValueError("--valid-memory must be the held-out valid partition")
        if set(e["source_sample_id"] for e in memory.index.values()) & set(e["source_sample_id"] for e in valid.index.values()):
            raise ValueError("Train and validation conversations overlap")
        config["model"]["checkpoint"] = str(Path(args.model).resolve())
        if args.backend == "verl":
            from training.verl_backend import VerlBackend
            runner = VerlBackend(args.model, config)
            updater = runner
        else:
            from training.hf_rl import HFUpdater
            runner = policy(args, config)
            updater = HFUpdater(runner, config)
        train_retrieval(config, memory, runner, updater, args.output, valid, args.resume)
    elif args.command == "collect-trees":
        config["model"]["checkpoint"] = str(Path(args.model).resolve())
        collect_forests(config, MemoryRun(args.memory, config), policy(args, config), args.output, resume=args.resume)
    elif args.command == "derive-sft":
        derive_sft(config, MemoryRun(args.memory, config), args.forests, args.output)
    elif args.command == "train-sft":
        from training.sft import train_sft
        config["model"]["checkpoint"] = str(Path(args.model).resolve())
        # torchrun ranks must share this directory; only rank zero creates its manifest.
        if int(os.environ.get("RANK", "0")) == 0:
            checked_output(args.output, resume=args.resume is not None)
            current_manifest = manifest(config, [args.selected])
            if args.resume:
                from experiments.io import read_json
                previous_manifest = read_json(Path(args.output) / "manifest.json")
                if any(previous_manifest[key] != current_manifest[key] for key in ("config", "inputs", "code_sha256")):
                    raise ValueError("SFT resume contract changed")
            if not args.resume:
                write_json(Path(args.output) / "manifest.json", current_manifest)
        train_sft(config, read_jsonl(args.selected), args.model, args.output, args.resume)
    elif args.command == "evaluate":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.serial_tool_execution and args.parallel_calls == 1:
            parser.error("--serial-tool-execution requires --parallel-calls greater than 1")
        checked_output(args.output, resume=args.resume)
        config["model"]["checkpoint"] = str(Path(args.model).resolve())
        config["evaluation"].update(max_parallel_calls=args.parallel_calls, limit=args.limit,
                                    serial_tool_execution=args.serial_tool_execution,
                                    inference_engine=args.inference_engine)
        memory = MemoryRun(args.memory, config, db_path=Path(args.output) / "db")
        # Resolve the backend's effective context budget before sealing a resume contract.
        from transformers import AutoConfig
        from models.hf import effective_prompt_limit
        effective_prompt_limit(config, AutoConfig.from_pretrained(args.model).max_position_embeddings)
        current = manifest(config, [memory.path / 'index.json', memory.path / 'questions.jsonl'])
        manifest_path = Path(args.output) / 'manifest.json'
        if args.resume and manifest_path.exists():
            from experiments.io import read_json
            previous = read_json(manifest_path)
            if any(previous[k] != current[k] for k in ('config', 'inputs', 'code_sha256')):
                raise ValueError('Evaluation resume contract changed')
        if args.resume and not config.get('runtime', {}).get('resumable_evaluation'):
            raise ValueError('Evaluation --resume requires resumable_evaluation')
        write_json(manifest_path, current)
        answers = 'answers.jsonl.gz' if config.get('runtime', {}).get('compress_traces') else 'answers.jsonl'
        result = evaluate(config, memory, policy(args, config), Path(args.output) / answers)
        write_json(Path(args.output) / "manifest.json", manifest(config, [memory.path / "index.json", memory.path / "questions.jsonl"]))
        print(result)
    elif args.command == "export-llamafactory":
        from training.llamafactory import export_selected
        export_selected(read_jsonl(args.selected), args.model, args.output, config)


if __name__ == "__main__":
    main()
