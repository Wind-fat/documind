.PHONY: help install seed ingest test lint fmt eval eval-smoke serve web demo verify

help:
	@echo "install  - uv sync --extra dev"
	@echo "ingest   - 灌入 3 份公开章程 + 生成评测集（首次约 15 分钟，含模型下载）"
	@echo "test     - pytest（离线：hash embedder + 词面重排）"
	@echo "lint     - ruff check"
	@echo "eval     - 四策略消融 + 阈值标定"
	@echo "serve    - 起 API :8080"
	@echo "verify   - lint + test，可粘贴运行"

install:
	uv sync --extra dev

seed:
	uv run documind seed

ingest:
	./.venv/Scripts/python.exe scripts/bootstrap_dev.py

test:
	./.venv/Scripts/python.exe -m pytest -q

lint:
	./.venv/Scripts/python.exe -m ruff check src tests scripts

fmt:
	./.venv/Scripts/python.exe -m ruff format src tests scripts

eval:
	./.venv/Scripts/python.exe scripts/run_ablation.py

eval-smoke:
	./.venv/Scripts/python.exe scripts/run_ablation.py --limit 12

serve:
	uv run uvicorn documind.app:app --host 127.0.0.1 --port 8080

web:
	cd web && npm run dev

verify: lint test
	@echo "OK: lint + 测试全绿"