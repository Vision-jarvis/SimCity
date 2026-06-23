.PHONY: install dev test test-all lint api frontend docker-up docker-down seed clean \
	train-pipeline retrain loadtest k8s-deploy k8s-delete

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

# ============ MLOps ============
train-pipeline:
	python run_training_pipeline.py train --demo

retrain:
	python run_training_pipeline.py schedule --observed-mae 0.55 --demo

# ============ Graph ============
graph-builder:
	python run_graph_builder.py

# ============ Testing ============
test:
	python -m pytest tests/ -q

test-all:
	python -m pytest tests/ api/tests/ graph/tests/ ingestion/tests/ nlp/tests/ -q

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

loadtest:
	locust -f scripts/locustfile.py --host http://localhost:8000

# ============ Kubernetes (k3s / OCI) ============
k8s-deploy:
	kubectl apply -k infra/k8s

k8s-delete:
	kubectl delete -k infra/k8s

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage simulation_cache
