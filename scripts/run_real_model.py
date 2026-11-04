"""Explicit real-API entry point; credentials stay in this process's memory."""

import argparse
from datetime import datetime
from getpass import getpass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="显式运行真实模型实验，会产生API费用；密钥不写入文件")
    parser.add_argument("--provider", choices=["qwen", "deepseek"], default="qwen")
    parser.add_argument("--config", type=Path, default=ROOT / "config/models.example.json")
    args = parser.parse_args()
    from dqagent.experiment import run_experiment
    from dqagent.providers import ChatCompletionsProvider
    try:
        provider = ChatCompletionsProvider.from_config(args.config, args.provider)
        print(f"将调用真实模型：{provider.model}。最多8轮模型请求，含有限HTTP重试。")
        if not os.environ.get(provider.api_key_env, "").strip():
            secret = getpass(f"请输入 {provider.api_key_env}（输入隐藏，仅当前进程使用）：").strip()
            if not secret:
                raise ValueError("没有输入密钥，未调用模型。")
            os.environ[provider.api_key_env] = secret
        output = ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + args.provider)
        result = run_experiment(ROOT / "data/prepared", output, mode="api", config=args.config, provider_profile=args.provider)
        print(json.dumps({"status": result["status"], "model_calls": result["agent"]["model_calls"], "output": str(output)}, ensure_ascii=False, indent=2))
        if result["status"] != "completed":
            raise SystemExit(1)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"未完成：{error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
