"""Student preparation and batch review, using only the bundled review core."""
from __future__ import annotations
import json
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .runtime import LocalQwenClient, ServiceFatalError
from .common import write_json
from .metrics import _audit_metrics
from .review.api import request_cache, stage_observer
from .review.cases import build_case, find_student_directories, open_readonly_database
from .review.pipeline import error_record, process_case
from .review.final_summary import polish_final_summary
from .review.exporter import export_run

def build_cases(ocr_root: Path, database: Path, people: list[dict], output: Path,
                ocr_records: dict[str, dict] | None = None):
    directories = find_student_directories(ocr_root)
    connection = open_readonly_database(database)
    cases = []
    errors = []
    try:
        for person in people:
            student_no = person["student_no"]
            ocr_record = (ocr_records or {}).get(student_no)
            if ocr_records is not None and (ocr_record is None or not ocr_record.get("success")):
                errors.append({
                    "student_no": student_no, "stage": "ocr", "error_type": "OCRFailed",
                    "message": (ocr_record or {}).get("error") or "该学生存在OCR失败材料，未进入审核",
                    "retry_count": 0,
                })
                continue
            if student_no not in directories:
                errors.append({
                    "student_no": student_no, "stage": "prepare", "error_type": "MissingOCR",
                    "message": "采样学生没有生成完整的OCR目录", "retry_count": 0,
                })
                continue
            case, case_errors = build_case(ocr_root, student_no, directories.get(student_no, []), connection)
            errors.extend(case_errors)
            if case is not None and not case_errors:
                cases.append(case)
    finally:
        connection.close()
    cases_path = output / "cases.jsonl"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_path.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    return cases, errors


def audit_cases(cases: list[dict], client: LocalQwenClient, output: Path, model_name: str, workers: int):
    results_path = output / "results.jsonl"
    errors_path = output / "errors.jsonl"
    events_path = output / "events.jsonl"
    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    results_stream = results_path.open("w", encoding="utf-8")
    errors_stream = errors_path.open("a", encoding="utf-8")
    events_stream = events_path.open("w", encoding="utf-8")
    people_stream = (output / "people.jsonl").open("a", encoding="utf-8")

    def append(stream, value):
        with lock:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()

    def run(case):
        student = case.get("student", {})
        no = str(student.get("student_no") or "")
        event_items = []
        stage_started = {}
        stage_seconds = {}
        request_counts = {}
        person_started = time.perf_counter()

        def observer(stage, state, **values):
            now = time.perf_counter()
            item = {"student_no": no, "stage": stage, "state": state,
                    "at": now, **{key: value for key, value in values.items() if key != "result"}}
            event_items.append(item)
            if state == "running":
                stage_started[stage] = now
            elif state in {"completed", "retry", "failed"} and stage in stage_started:
                stage_seconds[stage] = stage_seconds.get(stage, 0.0) + now - stage_started.pop(stage)
            if state in {"running", "retry", "failed", "completed"}:
                request_counts[stage] = request_counts.get(stage, 0) + (1 if state == "running" else 0)
            append(events_stream, item)
            if state == "running":
                print(f"[AUDIT][{no}] {stage} 开始", flush=True)
            elif state == "retry":
                print(f"[AUDIT][{no}] {stage} 重试，第{values.get('attempt', '?')}次", flush=True)

        try:
            observer("学生审核", "running")
            token = stage_observer.set(observer)
            cache_token = request_cache.set(None)
            try:
                result, gate_record, audit_record = process_case(
                    case, client, model_name, model_name, model_name
                )
                result["materials"] = case.get("materials", [])
                result, summary_record = polish_final_summary(result, client, model_name)
                status = (result.get("destination_verification") or {}).get("status")
                if gate_record is None:
                    observer("门控", "skipped", reason=status)
                if audit_record is None:
                    observer("审核", "skipped", reason=status)
                if summary_record.get("state") == "skipped":
                    observer("摘要", "skipped", reason=status)
                observer("学生审核", "completed")
            finally:
                request_cache.reset(cache_token)
                stage_observer.reset(token)
            result["benchmark_timing"] = {
                "student_seconds": time.perf_counter() - person_started,
                "stage_seconds": stage_seconds, "request_counts": request_counts,
                "event_count": len(event_items), "gate_record": gate_record,
                "audit_record": audit_record, "summary_record": summary_record,
            }
            append(results_stream, result)
            status = (result.get("destination_verification") or {}).get("status") or "success"
            usage = {
                "gate": (gate_record or {}).get("usage", {}),
                "audit": (audit_record or {}).get("usage", {}),
                "summary": (summary_record or {}).get("usage", {}),
            }
            return {"success": True, "student_no": no,
                    "seconds": result["benchmark_timing"]["student_seconds"],
                    "stage_seconds": stage_seconds, "request_counts": request_counts,
                    "usage": usage, "result_status": status,
                    "pages": result.get("material_summary", {}).get("page_count", 0)}
        except Exception as exc:
            observer("学生审核", "failed", message=str(exc))
            error = error_record(case, exc)
            append(errors_stream, error)
            if isinstance(client, LocalQwenClient):
                try:
                    client.ensure_healthy()
                except ServiceFatalError as fatal:
                    with lock:
                        write_json(output / 'fatal_error.json', {'stage': 'qwen_service', 'message': str(fatal)})
                    raise
            return {"success": False, "student_no": no,
                    "seconds": time.perf_counter() - person_started,
                    "error": error, "stage_seconds": stage_seconds}

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, case) for case in cases]
            records = []
            for future in as_completed(futures):
                try:
                    record = future.result()
                except ServiceFatalError:
                    for pending in futures:
                        pending.cancel()
                    raise
                records.append(record)
                append(people_stream, record)
                with lock:
                    completed += 1
                    if completed % 10 == 0 or completed == len(cases):
                        failures = sum(not item["success"] for item in records)
                        print(f"[AUDIT] {completed}/{len(cases)} 人完成，失败{failures}，已用时{time.perf_counter() - started:.1f}s", flush=True)
    finally:
        results_stream.close()
        errors_stream.close()
        events_stream.close()
        people_stream.close()
    elapsed = time.perf_counter() - started
    return records, elapsed, _audit_metrics(records, elapsed)


def export_results(output: Path):
    export_run(output / "results.jsonl", output / "errors.jsonl", output / "审核结果.xlsx")
