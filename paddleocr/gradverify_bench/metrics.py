"""Measured-stage statistics and Chinese benchmark report."""
from __future__ import annotations
import json
from pathlib import Path

def _percentile(values: list[float], probability: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _audit_metrics(records: list[dict], elapsed: float) -> dict:
    durations = [float(item["seconds"]) for item in records]
    successful = [item for item in records if item.get("success")]
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    stage_values = {}
    distribution = {"success": 0, "failed": 0, "manual_only": 0, "unmapped": 0}
    pages = 0
    for item in records:
        if not item.get("success"):
            distribution["failed"] += 1
        else:
            status = item.get("result_status") or "success"
            distribution[status] = distribution.get(status, 0) + 1
        pages += int(item.get("pages", 0) or 0)
        for stage, seconds in item.get("stage_seconds", {}).items():
            stage_values.setdefault(stage, []).append(float(seconds))
        for usage in item.get("usage", {}).values():
            if isinstance(usage, dict):
                for key in total_tokens:
                    total_tokens[key] += int(usage.get(key, 0) or 0)
    stage_statistics = {
        stage: {
            "mean_seconds": sum(values) / len(values),
            "median_seconds": _percentile(values, 0.50),
            "p90_seconds": _percentile(values, 0.90),
            "p95_seconds": _percentile(values, 0.95),
            "count": len(values),
        }
        for stage, values in stage_values.items()
    }
    total = total_tokens["total_tokens"]
    return {
        "people": len(records),
        "successful_people": len(successful),
        "failed_people": len(records) - len(successful),
        "duration_seconds": {
            "mean": sum(durations) / len(durations) if durations else None,
            "median": _percentile(durations, 0.50),
            "p90": _percentile(durations, 0.90),
            "p95": _percentile(durations, 0.95),
        },
        "pages": pages,
        "pages_per_hour": pages * 3600 / elapsed if elapsed else None,
        "people_per_hour": len(successful) * 3600 / elapsed if elapsed else None,
        "tokens": total_tokens,
        "tokens_per_second": total / elapsed if elapsed else None,
        "tokens_per_hour": total * 3600 / elapsed if elapsed else None,
        "stage_timing": stage_statistics,
        "outcome_distribution": distribution,
    }


def write_report(output: Path, summaries: list[dict], comparison: dict | None = None) -> None:
    lines = [
        "# GradVerify OCR＋本地 Qwen BF16 全流程测速报告",
        "",
        "本报告由批处理测速脚本自动生成。正式吞吐不包含服务启动和预热；完整墙钟包含 OCR、案件整理、Qwen 加载/切换、审核和导出。此轮不进行准确率评测。",
        "",
        "## 条件汇总",
        "",
        "| 条件 | 样本/案件 | OCR 人/小时 | OCR 页/小时 | 审核人/小时 | 审核页/小时 | token/秒 | 全流程人/小时 | GPU 小时/成功人 | 失败 | 人工/跳过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        metrics = item.get("audit_metrics", {})
        distribution = metrics.get("outcome_distribution", {})
        lines.append(
            "| {condition} | {sample}/{cases} | {ocr:.2f} | {ocr_pages:.2f} | {audit:.2f} | {pages:.2f} | {tokens:.2f} | {e2e:.2f} | {gpu:.4f} | {failed} | {skipped} |".format(
                condition=item.get("condition"), sample=item.get("sample_people", 0), cases=item.get("auditable_cases", 0),
                ocr=item.get("ocr_people_per_hour") or 0, audit=item.get("audit_people_per_hour") or 0,
                ocr_pages=item.get("ocr_pages_per_hour") or 0,
                pages=metrics.get("pages_per_hour") or 0, tokens=metrics.get("tokens_per_second") or 0,
                e2e=item.get("end_to_end_people_per_hour") or 0,
                gpu=item.get("gpu_hours_per_successful_person") or 0,
                failed=distribution.get("failed", item.get("failed_audits", 0)),
                skipped=distribution.get("manual_only", 0) + distribution.get("unmapped", 0),
            )
        )
    lines += ["", "## 分阶段耗时", ""]
    for item in summaries:
        lines.append(f"### {item.get('condition')}")
        lines.append("")
        lines.append(
            f"OCR 正式 {item.get('ocr_seconds') or 0:.2f}s，OCR 冷启动含预热 {item.get('ocr_cold_seconds') or 0:.2f}s；"
            f"案件整理 {item.get('case_prepare_seconds') or 0:.2f}s；"
            f"Qwen 启动 {item.get('qwen_startup_seconds') or 0:.2f}s，预热 {item.get('qwen_warmup_seconds') or 0:.2f}s，"
            f"审核正式 {item.get('audit_seconds') or 0:.2f}s，导出 {item.get('export_seconds') or 0:.2f}s。"
        )
        lines.append(f"Qwen GPU 利用率/显存采样：`{json.dumps(item.get('qwen_gpu_stats', {}), ensure_ascii=False)}`。")
        stage_timing = item.get("audit_metrics", {}).get("stage_timing", {})
        if stage_timing:
            lines.append("")
            lines.append("| 审核环节 | 平均(s) | 中位(s) | P90(s) | P95(s) | 请求人数 |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for stage, values in stage_timing.items():
                lines.append(
                    f"| {stage} | {values.get('mean_seconds', 0):.2f} | {values.get('median_seconds', 0):.2f} | "
                    f"{values.get('p90_seconds', 0):.2f} | {values.get('p95_seconds', 0):.2f} | {values.get('count', 0)} |"
                )
        lines.append("")
        lines.append(f"输入清单 SHA-256：`{item.get('input_manifest_sha256', '')}`；案件哈希：`{item.get('case_hash_sha256', '')}`；审核核心冻结副本 SHA-256：`{item.get('review_core_sha256', '')}`。")
    if comparison:
        lines += ["", "## 2GPU／4GPU 对照", "", f"```json\n{json.dumps(comparison, ensure_ascii=False, indent=2)}\n```"]
    lines += ["", "## 完整性", "", "每个条件应有 128 条学生终态记录；`student_terminal_gap` 必须为 0。若 GPU、服务或审核出现致命错误，脚本会停止该条件并保留日志。"]
    (output / "测速报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
