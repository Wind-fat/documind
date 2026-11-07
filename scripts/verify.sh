#!/usr/bin/env bash
# 一键验收：lint + 测试 + 检索门禁（Windows git-bash 可直接粘贴运行，不依赖 make）
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/Scripts/python.exe
if [ ! -x "$PY" ]; then PY=./.venv/bin/python; fi

echo "== ruff"
"$PY" -m ruff check src tests scripts || { echo "FAIL: ruff"; exit 1; }

echo "== pytest"
"$PY" -m pytest -q || { echo "FAIL: pytest"; exit 1; }

echo "== 检索门禁（离线：hash embedder）"
DOCUMIND_EMBED_BACKEND=hash DOCUMIND_RERANK_BACKEND=lexical DOCUMIND_DATABASE_URL="sqlite+aiosqlite:///./gate.db" \
  "$PY" scripts/ci_eval_gate.py || { echo "FAIL: eval gate"; exit 1; }

echo
echo "OK: lint + 测试 + 检索门禁全绿"