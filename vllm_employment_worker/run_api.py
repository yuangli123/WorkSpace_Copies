from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动就业佐证材料核查 API")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "api.example.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["EMPLOYMENT_API_CONFIG"] = str(args.config.resolve())
    import uvicorn

    # 后台队列保存在当前 API 进程中，只允许一个 worker。
    uvicorn.run(
        "employment_worker.api.app:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

