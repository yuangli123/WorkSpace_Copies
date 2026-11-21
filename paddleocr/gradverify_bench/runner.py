"""Command-line orchestration: OCR, cases, Qwen, audit, export, comparison."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from .benchmarks import benchmark_gpu_people as ocr_benchmark
from .ocr.vlm_service import gpu_statistics
from .common import write_json, model_hash, review_core_hash, visible_gpu_ids
from .runtime import QwenService, LocalQwenClient
from .metrics import write_report

def make_ocr_args(args, gpu_count: int, output: Path, gpu_ids: list[str]):
    return SimpleNamespace(
        input_root=args.input_root, output_root=output, model_root=args.ocr_model_root,
        runner=args.ocr_runner, vlm_runner=args.ocr_vlm_runner, profile=args.ocr_profile,
        preset=args.ocr_preset, file_batch_size=args.ocr_file_batch_size,
        service_timeout=args.ocr_service_timeout, cpu_threads=args.cpu_threads,
        timeout_seconds=args.timeout_seconds, gpus=list(range(1, gpu_count + 1)),
        gpu_ids=gpu_ids,
    )


def run_condition(args, gpu_ids: list[str], people: list[dict], warmup: list[dict], condition: Path):
    # Spawned OCR interpreters import the CLI; only the controller needs review dependencies.
    from .audit import build_cases, audit_cases, export_results

    condition.mkdir(parents=True, exist_ok=False)
    ocr_dir = condition / "ocr_phase"
    ocr_args = make_ocr_args(args, len(gpu_ids), ocr_dir, gpu_ids)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    try:
        ocr_started = time.perf_counter()
        ocr_summary = ocr_benchmark.run_case(ocr_args, gpu_ids, people, warmup, ocr_dir)
        ocr_seconds = time.perf_counter() - ocr_started
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
    ocr_root = ocr_dir / "ocr"
    ocr_records = {}
    people_path = ocr_dir / "people.jsonl"
    if people_path.is_file():
        with people_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    ocr_records[str(record.get("student_no") or "")] = record
    case_prepare_started = time.perf_counter()
    cases, preparation_errors = build_cases(
        ocr_root, args.database, people, condition / "audit_phase", ocr_records
    )
    case_prepare_seconds = time.perf_counter() - case_prepare_started
    case_hash = hashlib.sha256(json.dumps([
        {"custom_id": case.get("custom_id"), "student": case.get("student"),
         "materials": case.get("materials", [])}
        for case in cases
    ], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    prep_error_path = condition / "audit_phase" / "errors.jsonl"
    prep_error_path.parent.mkdir(parents=True, exist_ok=True)
    with prep_error_path.open("w", encoding="utf-8") as stream:
        for error in preparation_errors:
            stream.write(json.dumps(error, ensure_ascii=False) + "\n")
    prepared_failures = sorted({str(error['student_no']) for error in preparation_errors})
    with (condition / 'audit_phase' / 'people.jsonl').open('w', encoding='utf-8') as stream:
        for student_no in prepared_failures:
            stream.write(json.dumps({'student_no': student_no, 'success': False,
                                     'stage': 'prepare'}, ensure_ascii=False) + '\n')
    if not cases:
        raise RuntimeError(f"No auditable cases were built for {condition.name}")
    qwen_started = time.perf_counter()
    qwen = QwenService(
        args.qwen_runner, args.qwen_model, args.qwen_model_name, gpu_ids,
        condition / "qwen_phase", args.qwen_max_model_len, max(4, args.audit_workers),
        args.qwen_gpu_memory_utilization, args.qwen_service_timeout, args.qwen_prefix_caching,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    qwen_monitor = None
    qwen_log = None
    qwen_gpu_path = condition / "qwen_phase" / "gpu.csv"
    try:
        with qwen as service:
            try:
                qwen_gpu_path.parent.mkdir(parents=True, exist_ok=True)
                qwen_log = qwen_gpu_path.open("w", encoding="utf-8")
                qwen_monitor = subprocess.Popen([
                    "nvidia-smi", "-i", ",".join(gpu_ids),
                    "--query-gpu=timestamp,index,uuid,name,driver_version,utilization.gpu,utilization.memory,"
                    "memory.used,memory.total,power.draw", "--format=csv", "-l", "1",
                ], stdout=qwen_log, stderr=subprocess.STDOUT)
            except OSError:
                if qwen_log is not None:
                    qwen_log.close()
                qwen_log = None
            client = LocalQwenClient(service.url)
            try:
                qwen_warmup_seconds = client.warmup(
                    args.qwen_model_name, condition / "qwen_phase" / "warmup.json"
                )
                audit_records, audit_seconds, audit_metrics = audit_cases(
                    cases, client, condition / "audit_phase", args.qwen_model_name, args.audit_workers
                )
            finally:
                client.close()
            if qwen_monitor is not None:
                qwen_monitor.terminate()
                qwen_monitor.wait(timeout=10)
                qwen_monitor = None
            if qwen_log is not None:
                qwen_log.close()
                qwen_log = None
            qwen_startup_seconds = service.startup_seconds
    finally:
        if qwen_monitor is not None:
            qwen_monitor.terminate()
            qwen_monitor.wait(timeout=10)
        if qwen_log is not None:
            qwen_log.close()
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
    export_started = time.perf_counter()
    export_results(condition / "audit_phase")
    export_seconds = time.perf_counter() - export_started
    elapsed = time.perf_counter() - ocr_started
    qwen_seconds = time.perf_counter() - qwen_started
    successful_people = audit_metrics["successful_people"]
    summary = {
        "condition": condition.name, "gpu_count": len(gpu_ids), "gpu_ids": gpu_ids,
        "sample_people": len(people), "auditable_cases": len(cases),
        "preparation_errors": len(preparation_errors),
        "student_terminal_records": len(audit_records) + len(prepared_failures),
        "student_terminal_expected": len(people),
        "student_terminal_gap": len(people) - len(audit_records) - len(prepared_failures),
        "input_manifest_sha256": args.input_manifest_sha256, "case_hash_sha256": case_hash,
        "review_core_sha256": args.review_core_sha256,
        "ocr_phase_wall_seconds": ocr_seconds,
        "ocr_seconds": ocr_summary.get("wall_seconds"),
        "ocr_cold_seconds": ocr_summary.get("cold_wall_seconds"),
        "case_prepare_seconds": case_prepare_seconds,
        "qwen_startup_seconds": qwen_startup_seconds, "audit_seconds": audit_seconds,
        "audit_metrics": audit_metrics,
        "qwen_warmup_seconds": qwen_warmup_seconds,
        "qwen_phase_seconds": qwen_seconds, "qwen_gpu_stats": gpu_statistics(qwen_gpu_path),
        "export_seconds": export_seconds, "wall_seconds": elapsed,
        "ocr_people_per_hour": ocr_summary.get("successful_people", 0) * 3600 / ocr_summary["wall_seconds"]
        if ocr_summary.get("wall_seconds") else None,
        "ocr_pages_per_hour": ocr_summary.get("successful_pages", 0) * 3600 / ocr_summary["wall_seconds"]
        if ocr_summary.get("wall_seconds") else None,
        "audit_people_per_hour": successful_people * 3600 / audit_seconds if audit_seconds else None,
        "end_to_end_people_per_hour": successful_people * 3600 / elapsed if elapsed else None,
        "gpu_hours_per_successful_person": len(gpu_ids) * elapsed / 3600 / successful_people
        if successful_people else None,
        "successful_audits": sum(item["success"] for item in audit_records),
        "failed_audits": sum(not item["success"] for item in audit_records),
        "ocr_summary": ocr_summary, "model": args.qwen_model_name,
        "model_sha256": args.qwen_model_sha256, "qwen_workers": args.audit_workers,
    }
    write_json(condition / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--ocr-model-root", type=Path, required=True)
    parser.add_argument("--ocr-runner", type=Path, required=True)
    parser.add_argument("--ocr-vlm-runner", type=Path)
    parser.add_argument("--ocr-profile", choices=("baseline", "native", "vllm"), default="vllm")
    parser.add_argument("--ocr-preset", choices=("conservative", "balanced", "throughput"), default="conservative")
    parser.add_argument("--ocr-file-batch-size", type=int, default=4)
    parser.add_argument("--qwen-runner", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--qwen-model-name", default="Qwen3.8-27B")
    parser.add_argument("--qwen-max-model-len", type=int, default=262144)
    parser.add_argument("--qwen-prefix-caching", action="store_true")
    parser.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--qwen-service-timeout", type=float, default=900)
    parser.add_argument("--ocr-service-timeout", type=float, default=600)
    parser.add_argument("--audit-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--people", type=int, default=128)
    parser.add_argument("--warmup-people", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=36000)
    parser.add_argument("--conditions", type=int, nargs="+", choices=(2, 4), default=[2, 4])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.audit_workers < 1 or args.cpu_threads < 1:
        raise SystemExit("--audit-workers and --cpu-threads must be positive")
    if args.ocr_profile == "vllm" and args.ocr_vlm_runner is None:
        raise SystemExit("--ocr-vlm-runner is required with --ocr-profile vllm")
    for path, label in ((args.input_root, "--input-root"), (args.database, "--database"),
                        (args.sample_manifest, "--sample-manifest"), (args.ocr_model_root, "--ocr-model-root"),
                        (args.ocr_runner, "--ocr-runner"), (args.qwen_runner, "--qwen-runner"),
                        (args.qwen_model, "--qwen-model")):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")
    for path, label in ((args.input_root, "--input-root"), (args.ocr_model_root, "--ocr-model-root"),
                        (args.qwen_model, "--qwen-model")):
        if not path.is_dir():
            raise SystemExit(f"{label} must be a directory: {path}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("--output-root must be a new or empty directory")
    args.output_root.mkdir(parents=True, exist_ok=True)
    gpu_ids = visible_gpu_ids(max(args.conditions))
    people, warmup, unmatched, fingerprints = ocr_benchmark.load_sample_manifest(
        args.sample_manifest, args.input_root.resolve(), args.people, args.warmup_people
    )
    args.qwen_model_sha256 = model_hash(args.qwen_model)
    args.review_core_sha256 = review_core_hash(Path(__file__).resolve().parent)
    args.input_manifest_sha256 = hashlib.sha256(json.dumps(
        fingerprints, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")).hexdigest()
    write_json(args.output_root / "manifest.json", {
        "people": people, "warmup": warmup, "unmatched_files": unmatched,
        "inputs": fingerprints, "input_manifest_sha256": args.input_manifest_sha256,
        "conditions": args.conditions,
        "qwen_model": str(args.qwen_model), "qwen_model_sha256": args.qwen_model_sha256,
        "review_core_sha256": args.review_core_sha256,
    })
    summaries = []
    for count in args.conditions:
        print(f"[{count}GPU] 开始 OCR 阶段", flush=True)
        summaries.append(run_condition(args, gpu_ids[:count], people, warmup,
                                       args.output_root / f"condition_{count}gpu"))
        write_json(args.output_root / "summary.json", summaries)
    comparison = None
    if len(summaries) >= 2:
        baseline = next((item for item in summaries if item["gpu_count"] == 2), None)
        four_gpu = next((item for item in summaries if item["gpu_count"] == 4), None)
        if baseline and four_gpu:
            baseline_rate = baseline.get("end_to_end_people_per_hour")
            comparison_rate = four_gpu.get("end_to_end_people_per_hour")
            comparison = {
                "input_manifest_sha256_equal": baseline.get("input_manifest_sha256") == four_gpu.get("input_manifest_sha256"),
                "case_hash_sha256_equal": baseline.get("case_hash_sha256") == four_gpu.get("case_hash_sha256"),
                "end_to_end_speedup_4gpu_over_2gpu": comparison_rate / baseline_rate
                if baseline_rate and comparison_rate else None,
                "ocr_speedup_4gpu_over_2gpu": four_gpu.get("ocr_people_per_hour") / baseline.get("ocr_people_per_hour")
                if baseline.get("ocr_people_per_hour") and four_gpu.get("ocr_people_per_hour") else None,
                "audit_speedup_4gpu_over_2gpu": four_gpu.get("audit_people_per_hour") / baseline.get("audit_people_per_hour")
                if baseline.get("audit_people_per_hour") and four_gpu.get("audit_people_per_hour") else None,
                "gpu_hours_per_person": {
                    "2gpu": baseline.get("gpu_hours_per_successful_person"),
                    "4gpu": four_gpu.get("gpu_hours_per_successful_person"),
                },
            }
            write_json(args.output_root / "comparison.json", comparison)
    with (args.output_root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_report(args.output_root, summaries, comparison)
    print(f"完成，报告目录：{args.output_root}", flush=True)
    return 1 if any(item["failed_audits"] for item in summaries) else 0
