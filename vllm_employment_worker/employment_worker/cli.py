from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .client import VLLMClient
from .config import RuntimeConfig, load_json_object
from .metadata import load_metadata
from .pipeline import check_dependencies, run_pipeline
from .prompts import PromptBuilder
from .schema import load_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "runtime.example.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过 vLLM 批量核查高校毕业生就业佐证材料"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, help="单个材料文件或材料根目录")
    parser.add_argument("--output-dir", type=Path, help="结果输出目录")
    parser.add_argument("--metadata", type=Path, help="登记字段 CSV/JSON/JSONL")
    parser.add_argument("--resume", action="store_true", help="按文件指纹续跑")
    parser.add_argument(
        "--retry-model-failures",
        action="store_true",
        help="与 --resume 配合，重新处理此前已人工兜底的模型失败文件",
    )
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件；0 为全部")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--check-dependencies", action="store_true")
    parser.add_argument("--check-server", action="store_true")
    parser.add_argument("--print-prompt", action="store_true")

    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--pages-per-request", type=int)
    parser.add_argument("--max-tokens", type=int, help="常规分段 OCR 输出上限")
    parser.add_argument(
        "--retry-max-tokens",
        type=int,
        help="单页长输出及合并失败时的重试上限",
    )
    parser.add_argument("--merge-max-tokens", type=int, help="文档合并输出上限")
    parser.add_argument(
        "--max-inference-attempts-per-document",
        type=int,
        help="每份文件最多发起的 vLLM 请求数（包含重试、拆页和合并）",
    )
    parser.add_argument(
        "--inference-max-pixels",
        type=int,
        help="送入 vLLM 的单页图像最大像素数；原始预处理存档不受影响",
    )
    parser.add_argument("--image-transport", choices=("file", "base64"))
    parser.add_argument("--libreoffice-bin")
    parser.add_argument("--pdftoppm-bin")
    parser.add_argument("--pdf-password")
    parser.add_argument("--pdf-password-file")
    parser.add_argument("--cache-dir", help="兼容旧名称：预处理页面缓存目录")
    parser.add_argument(
        "--archive-dir",
        help="持久化预处理存档目录；优先于 --cache-dir",
    )
    parser.add_argument(
        "--no-preprocess-progress",
        action="store_true",
        help="关闭预处理进度条",
    )
    parser.add_argument("--save-raw-responses", action="store_true")
    parser.add_argument("--disable-json-schema", action="store_true")
    return parser.parse_args()


def apply_overrides(config: RuntimeConfig, args: argparse.Namespace) -> None:
    for argument, field_name in (
        (args.api_base, "api_base"),
        (args.model, "model"),
        (args.concurrency, "concurrency"),
        (args.pages_per_request, "pages_per_request"),
        (args.max_tokens, "max_tokens"),
        (args.retry_max_tokens, "retry_max_tokens"),
        (args.merge_max_tokens, "merge_max_tokens"),
        (
            args.max_inference_attempts_per_document,
            "max_inference_attempts_per_document",
        ),
        (args.inference_max_pixels, "inference_max_pixels"),
        (args.image_transport, "image_transport"),
        (args.libreoffice_bin, "libreoffice_bin"),
        (args.pdftoppm_bin, "pdftoppm_bin"),
        (args.pdf_password, "pdf_password"),
        (args.pdf_password_file, "pdf_password_file"),
        (args.cache_dir, "cache_dir"),
    ):
        if argument is not None:
            setattr(config, field_name, argument)
    if args.archive_dir is not None:
        config.cache_dir = args.archive_dir
    if args.no_preprocess_progress:
        config.show_preprocess_progress = False
    if os.environ.get("EMPLOYMENT_PDF_PASSWORD") and args.pdf_password is None:
        config.pdf_password = os.environ["EMPLOYMENT_PDF_PASSWORD"]
    if args.save_raw_responses:
        config.save_raw_responses = True
    if args.disable_json_schema:
        config.use_json_schema = False
    config.validate()


async def _check_server(config: RuntimeConfig) -> list[str]:
    schema = load_schema(config.output_schema_file)
    prompts = PromptBuilder(config.prompt_dir, load_json_object(config.business_rules_file))
    client = VLLMClient(config, schema)
    client.system_prompt = prompts.system_prompt
    try:
        return await client.check_server()
    finally:
        await client.close()


def main() -> int:
    args = parse_args()
    config = RuntimeConfig.from_file(args.config)
    apply_overrides(config, args)

    if args.check_dependencies:
        print(json.dumps(check_dependencies(config), ensure_ascii=False, indent=2))
        return 0

    if args.print_prompt:
        prompts = PromptBuilder(config.prompt_dir, load_json_object(config.business_rules_file))
        print(prompts.system_prompt)
        print("\n===== USER PROMPT =====\n")
        print(
            prompts.build_review(
                relative_path="示例/就业协议.pdf",
                page_numbers=[1, 2],
                registered_fields={},
            )
        )
        return 0

    if args.check_server:
        models = asyncio.run(_check_server(config))
        print(json.dumps({"api_base": config.api_base, "models": models}, ensure_ascii=False, indent=2))
        return 0

    if args.input is None or args.output_dir is None:
        raise SystemExit("正式运行必须同时提供 --input 和 --output-dir")
    if not args.input.exists():
        raise SystemExit(f"输入路径不存在：{args.input}")
    if args.limit < 0:
        raise SystemExit("--limit 不能小于 0")

    metadata = load_metadata(args.metadata)
    summary = asyncio.run(
        run_pipeline(
            config=config,
            input_path=args.input,
            output_dir=args.output_dir,
            metadata=metadata,
            resume=args.resume,
            limit=args.limit,
            preprocess_only=args.preprocess_only,
            retry_model_failures=args.retry_model_failures,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
