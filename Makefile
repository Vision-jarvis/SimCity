.PHONY: install dev test lint api frontend docker-up docker-down seed clean

# ============ Setup ============
install:
	python -m venv .venv
	.venv/Scripts/activate && pip install -r requirements.txt
	cd frontend && npm install

# ============ Development ============
dev: docker-up api frontend

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

# ============ Infrastructure ============
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-prod:
	docker-compose -f docker-compose.prod.yml up -d

# ============ Data ============
seed:
	python scripts/seed_graph.py

generate-data:
	python data/synthetic_generator.py --events 10000 --out data/synthetic_events.pkl

backfill:
	python scripts/backfill_gdelt.py

# ============ ML ============
train:
	python train.py

evaluate:
	python evaluate.py

dry-run:
	python dry_run.py

hawkes-baseline:
	python run_hawkes_baseline.py --data data/synthetic_events.pkl --epochs 8

cascade-monitor:
	python run_cascade_monitor.py --data data/synthetic_events.pkl --epochs 4

# ============ Graph ============
graph-builder:
	python run_graph_builder.py

# ============ Testing ============
test:
	python -m pytest tests/ -q

test-cov:
	python -m pytest tests/ --cov=. --cov-report=html

lint:
	python -m ruff check .
	python -m mypy . --ignore-missing-imports

# ============ Utilities ============
smoke:
	python scripts/smoke_test.py

benchmark:
	python scripts/benchmark.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage simulation_cache
