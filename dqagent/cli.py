"""Small command line entry point. No network or model calls by default."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="数据质量处理智能体：构造、离线运行、真实API实验")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="从UCI下载公开源数据，需要网络")
    download.add_argument("--output", type=Path, default=Path("data/raw"))
    prepare = sub.add_parser("prepare", help="按固定种子划分并构造四类训练数据问题")
    prepare.add_argument("--raw", type=Path, default=Path("data/raw/bank-full.csv"))
    prepare.add_argument("--output", type=Path, default=Path("data/prepared"))
    prepare.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("run", help="对比基线与智能体；默认mock不调用模型")
    run.add_argument("--data", type=Path, default=Path("data/prepared"))
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--mode", choices=["mock", "api"], default="mock")
    run.add_argument("--provider", choices=["qwen", "deepseek"], default="qwen")
    run.add_argument("--config", type=Path, default=Path("config/models.example.json"))
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--compare", action="store_true", help="加入原方案、单项改动和等预算预设搜索对照")
    run.add_argument("--guided", action=argparse.BooleanOptionalAction, default=None,
                     help="按训练画像指导候选并统一验证阈值。API 模式默认开启")
    args = parser.parse_args()
    try:
        if args.command == "download":
            from .data import download_bank
            print(download_bank(args.output))
        elif args.command == "prepare":
            from .data import prepare_dataset
            result = prepare_dataset(args.raw, args.output, seed=args.seed)
            print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
        else:
            from .experiment import run_experiment
            output = args.output or Path("runs") / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + args.mode)
            print("离线规则模拟：不调用大模型。" if args.mode == "mock" else "真实API模式：将调用所选模型并产生费用。")
            result = run_experiment(args.data, output, mode=args.mode, config=args.config,
                                    provider_profile=args.provider, seed=args.seed, compare=args.compare,
                                    guided=args.guided)
            print(json.dumps({"status": result["status"], "output": str(output), "model_calls": result["agent"]["model_calls"]}, ensure_ascii=False, indent=2))
            if result["status"] != "completed":
                raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as error:
        # API provider errors contain fixed safe descriptions, never raw HTTP bodies.
        print(f"运行未完成：{error}", file=sys.stderr)
        raise SystemExit(2) from None
