.PHONY: install test lint data train serve clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=src --cov-report=term-missing -v

lint:
	ruff check src/ tests/

data:
	python scripts/download_data.py

train:
	python scripts/run_experiment.py --config configs/default.yaml

serve:
	uvicorn src.api.app:app --reload --port 8000

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
