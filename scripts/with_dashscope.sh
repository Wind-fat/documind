#!/usr/bin/env bash
# 用本机 magpie 的 "Qwen (China)" 凭据，把 DocuMind 指到真实的 DashScope compatible-mode。
#
#   scripts/with_dashscope.sh ./.venv/Scripts/python.exe -m uvicorn documind.app:app --port 8913
#   scripts/with_dashscope.sh ./.venv/Scripts/python.exe -m documind.cli ask "公司注册资本是多少"
#
# 为什么是一个脚本而不是 .env：
# - key 只在**运行时**从 ~/.config/magpie/providers.json 读进子进程环境变量，
#   不落盘、不打印、不进仓库、不进 shell history（命令替换的内容不会显示）；
# - 所以 `git status` 永远不会看到凭据，报告里也不会有。
#
# 可覆盖：MAGPIE_PROVIDERS（providers.json 路径）、DOCUMIND_PROVIDER_NAME（条目名）、
#         DOCUMIND_LLM_MODEL / DOCUMIND_EMBED_API_MODEL / DOCUMIND_EMBED_API_DIM / DOCUMIND_EMBED_BACKEND。
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "用法: scripts/with_dashscope.sh <命令...>" >&2
  exit 64
fi

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/Scripts/python.exe ]; then PY=./.venv/Scripts/python.exe
  elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
  else PY=python3
  fi
fi

# 命令替换：脚本内容不会出现在终端，也不会进日志
eval "$("$PY" - <<'PYEOF'
import json, os, shlex, sys

path = os.path.expanduser(os.environ.get("MAGPIE_PROVIDERS", "~/.config/magpie/providers.json"))
name = os.environ.get("DOCUMIND_PROVIDER_NAME", "Qwen (China)")
try:
    data = json.load(open(path, encoding="utf-8"))
except OSError as exc:
    sys.exit(f"读不到 providers.json（{exc.__class__.__name__}）：{path}")

item = next((p for p in data.get("providers", []) if p.get("name") == name), None)
if not item or not item.get("chat") or not item.get("key"):
    sys.exit(f"providers.json 里没有可用的条目：{name!r}")

base = item["chat"].rstrip("/")
env = {
    "DOCUMIND_LLM_BASE_URL": base,
    "DOCUMIND_LLM_API_KEY": item["key"],
    "DOCUMIND_LLM_MODEL": os.environ.get("DOCUMIND_LLM_MODEL") or "qwen3.8-flash",
    "DOCUMIND_EMBED_BACKEND": os.environ.get("DOCUMIND_EMBED_BACKEND") or "dashscope",
    "DOCUMIND_EMBED_API_BASE": base,
    "DOCUMIND_EMBED_API_KEY": item["key"],
    "DOCUMIND_EMBED_API_MODEL": os.environ.get("DOCUMIND_EMBED_API_MODEL") or "text-embedding-v3",
    "DOCUMIND_EMBED_API_DIM": os.environ.get("DOCUMIND_EMBED_API_DIM") or "1024",
}
for key, value in env.items():
    print(f"export {key}={shlex.quote(value)}")
PYEOF
)"

exec "$@"