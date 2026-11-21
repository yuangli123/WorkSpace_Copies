from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _secure(path: Path, directory: bool = False) -> None:
    if os.name == "nt":
        return
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError:
        pass


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    _secure(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    _secure(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                values.append(item)
    return values


class ResultStore:
    def __init__(self, output_dir: Path, save_raw: bool) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _secure(self.output_dir, directory=True)
        self.results_path = self.output_dir / "document_results.jsonl"
        self.errors_path = self.output_dir / "errors.jsonl"
        self.chunks_dir = self.output_dir / "chunk_results"
        self.save_raw = save_raw
        if save_raw:
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            _secure(self.chunks_dir, directory=True)

    def completed_source_ids(self, include_model_failures: bool = True) -> set[str]:
        completed_statuses = {"success"}
        if include_model_failures:
            completed_statuses.add("manual_review_fallback")
        return {
            str(item.get("source", {}).get("source_id"))
            for item in read_jsonl(self.results_path)
            if item.get("status") in completed_statuses
            and item.get("source", {}).get("source_id")
        }

    def append_result(self, record: dict[str, Any]) -> None:
        _append_jsonl(self.results_path, record)

    def append_error(self, error: dict[str, Any]) -> None:
        payload = {
            "occurred_at": datetime.now().isoformat(timespec="seconds"),
            **error,
        }
        _append_jsonl(self.errors_path, payload)

    def write_chunks(self, source_id: str, payloads: list[dict[str, Any]]) -> None:
        if not self.save_raw:
            return
        _write_json(self.chunks_dir / f"{source_id}.json", payloads)

    def write_skipped(self, skipped: Iterable[Path]) -> None:
        _write_json(
            self.output_dir / "skipped_files.json",
            [str(path.resolve()) for path in skipped],
        )

    def write_run_config(self, value: dict[str, Any]) -> None:
        _write_json(self.output_dir / "run_config.json", value)

    def write_preprocess_manifest(self, documents: Iterable[Any]) -> Path:
        path = self.output_dir / "preprocess_manifest.json"
        manifest = [
            {
                "source_id": item.source_id,
                "source_file": str(item.source_file),
                "relative_path": item.relative_path,
                "format": item.source_format,
                "from_archive": item.from_cache,
                "page_count": len(item.pages),
                "pages": [
                    {"page": page.page_number, "image_path": str(page.image_path)}
                    for page in item.pages
                ],
            }
            for item in documents
        ]
        _write_json(path, manifest)
        return path

    def finalize(
        self,
        preprocess_error_count: int,
        skipped_count: int,
        archive_hits: int,
        archive_created: int,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        latest = self._latest_records()
        records = list(latest.values())
        self._write_summary_csv(records)
        all_errors = read_jsonl(self.errors_path)
        successful_source_ids = {
            str(item.get("source", {}).get("source_id", ""))
            for item in records
            if item.get("status") == "success"
            and item.get("source", {}).get("source_id")
        }
        unresolved_error_keys = {
            str(item.get("source_id") or item.get("relative_path") or item.get("source_file"))
            for item in all_errors
            if str(item.get("source_id", "")) not in successful_source_ids
        }
        unresolved_error_keys.discard("")
        categories = Counter(
            str(item.get("result", {}).get("document_classification", {}).get("category", "无法判断"))
            for item in records
        )
        decisions = Counter(
            str(item.get("result", {}).get("decision", {}).get("recommended_result", "人工复核"))
            for item in records
        )
        summary = {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "completed_documents": len(records),
            "successful_documents": sum(
                item.get("status") == "success" for item in records
            ),
            "model_failure_fallback_documents": sum(
                item.get("status") == "manual_review_fallback" for item in records
            ),
            "error_records": len(unresolved_error_keys),
            "unresolved_error_documents": len(unresolved_error_keys),
            "historical_error_records": len(all_errors),
            "preprocess_error_count": preprocess_error_count,
            "skipped_unsupported_count": skipped_count,
            "preprocess_archive_hits": archive_hits,
            "preprocess_archive_created": archive_created,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "category_counts": dict(categories),
            "decision_counts": dict(decisions),
            "outputs": {
                "jsonl": str(self.results_path),
                "csv": str(self.output_dir / "document_summary.csv"),
                "errors": str(self.errors_path),
            },
        }
        _write_json(self.output_dir / "run_summary.json", summary)
        return summary

    def _latest_records(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for item in read_jsonl(self.results_path):
            relative_path = str(item.get("source", {}).get("relative_path", ""))
            if relative_path and item.get("status") in {
                "success",
                "manual_review_fallback",
            }:
                latest[relative_path] = item
        return latest

    def _write_summary_csv(self, records: list[dict[str, Any]]) -> None:
        path = self.output_dir / "document_summary.csv"
        fieldnames = [
            "relative_path",
            "processing_status",
            "source_format",
            "page_count",
            "category",
            "subcategory",
            "document_name",
            "classification_confidence",
            "recommended_result",
            "summary",
            "person_name",
            "organization_name",
            "unified_social_credit_code",
            "signing_date",
            "manual_review_items",
            "missing_items",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in sorted(records, key=lambda item: item["source"]["relative_path"]):
                result = record.get("result", {})
                classification = result.get("document_classification", {})
                decision = result.get("decision", {})
                fields = self._field_map(result.get("key_fields", []))
                writer.writerow(
                    {
                        "relative_path": record["source"].get("relative_path", ""),
                        "processing_status": record.get("status", ""),
                        "source_format": record["source"].get("format", ""),
                        "page_count": record["source"].get("page_count", 0),
                        "category": classification.get("category", ""),
                        "subcategory": classification.get("subcategory", ""),
                        "document_name": classification.get("document_name", ""),
                        "classification_confidence": classification.get("confidence", ""),
                        "recommended_result": decision.get("recommended_result", ""),
                        "summary": decision.get("summary", ""),
                        "person_name": self._first(fields, "姓名", "毕业生姓名", "学生姓名", "乙方姓名"),
                        "organization_name": self._first(fields, "单位名称", "用人单位名称", "甲方名称", "公司名称"),
                        "unified_social_credit_code": self._first(fields, "统一社会信用代码"),
                        "signing_date": self._first(fields, "签订日期", "签署日期", "日期"),
                        "manual_review_items": "；".join(decision.get("manual_review_items", [])),
                        "missing_items": "；".join(decision.get("missing_items", [])),
                    }
                )
        _secure(path)

    @staticmethod
    def _field_map(fields: Any) -> dict[str, str]:
        output: dict[str, str] = {}
        if not isinstance(fields, list):
            return output
        for item in fields:
            if isinstance(item, dict) and item.get("field_name"):
                output[str(item["field_name"])] = str(item.get("value", ""))
        return output

    @staticmethod
    def _first(values: dict[str, str], *names: str) -> str:
        return next((values[name] for name in names if values.get(name)), "")
