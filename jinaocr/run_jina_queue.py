#!/usr/bin/env python3
"""Assign top-level categories to all GPUs visible in one Slurm job."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
from pathlib import Path

from run_jina_category import DEFAULT_MODEL_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--runner", type=Path,
        default=Path(__file__).resolve().with_name("run_jina_env.sh"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    model_dir = args.model_dir.resolve()
    runner = args.runner.resolve()
    category_script = Path(__file__).resolve().with_name("run_jina_category.py")
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if (input_root == output_root or input_root in output_root.parents
            or output_root in input_root.parents):
        raise SystemExit("Input and output roots must not overlap")
    if not runner.is_file():
        raise SystemExit(f"GPU environment runner does not exist: {runner}")
    if not category_script.is_file():
        raise SystemExit(f"Category script does not exist: {category_script}")
    if not (model_dir / "REVISION").is_file():
        raise SystemExit(f"Model is not prepared: {model_dir}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("CUDA_VISIBLE_DEVICES is empty; run inside a GPU srun job")
    categories = sorted(path.name for path in input_root.iterdir() if path.is_dir())
    if not categories:
        raise SystemExit(f"No top-level categories found under: {input_root}")

    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    log_dir = category_script.parent / "logs" / f"jina_queue_{job_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    cpu_total = int(os.environ.get("SLURM_CPUS_PER_TASK", len(gpu_ids)))
    cpu_per_worker = max(1, cpu_total // len(gpu_ids))
    work: queue.Queue[tuple[int, str]] = queue.Queue()
    for number, category in enumerate(categories, 1):
        work.put((number, category))
    failures: list[str] = []
    failure_lock = threading.Lock()

    def worker(slot: int, gpu_id: str) -> None:
        while True:
            try:
                number, category = work.get_nowait()
            except queue.Empty:
                return

            prefix = f"{number:03d}_{category}"
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=gpu_id,
                OMP_NUM_THREADS=str(cpu_per_worker),
                MKL_NUM_THREADS=str(cpu_per_worker),
                HF_HUB_OFFLINE="1",
            )
            command = [
                "bash", str(runner), str(category_script),
                "--input-root", str(input_root),
                "--output-root", str(output_root),
                "--model-dir", str(model_dir),
                "--category", category,
            ]
            if args.force:
                command.append("--force")
            print(f"[GPU {slot}] START {category}", flush=True)
            try:
                with (log_dir / f"{prefix}.out").open("w", encoding="utf-8") as stdout, \
                     (log_dir / f"{prefix}.err").open("w", encoding="utf-8") as stderr:
                    result = subprocess.run(
                        command, env=env, stdout=stdout, stderr=stderr, check=False
                    )
                if result.returncode:
                    with failure_lock:
                        failures.append(category)
                    print(f"[GPU {slot}] FAIL  {category} (exit {result.returncode})", flush=True)
                else:
                    print(f"[GPU {slot}] DONE  {category}", flush=True)
            except OSError as error:
                with failure_lock:
                    failures.append(category)
                print(f"[GPU {slot}] FAIL  {category}: {error}", flush=True)
            finally:
                work.task_done()

    workers = [
        threading.Thread(target=worker, args=(slot, gpu_id))
        for slot, gpu_id in enumerate(gpu_ids)
    ]
    print(f"GPUs: {visible}")
    print(f"Categories: {len(categories)}")
    print(f"Logs: {log_dir}", flush=True)
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    print(f"Finished: {len(categories) - len(failures)} succeeded, {len(failures)} failed")
    if failures:
        print("Failed categories: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
