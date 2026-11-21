#!/usr/bin/env python3
"""Standalone batch benchmark entry point; all application code lives in this directory."""
from gradverify_bench.runner import main, parse_args
from gradverify_bench.common import visible_gpu_ids
from gradverify_bench.runtime import LocalQwenClient, QwenService
from gradverify_bench.metrics import _audit_metrics, write_report

if __name__ == "__main__":
    raise SystemExit(main())
