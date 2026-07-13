.DEFAULT_GOAL := help

UV ?= uv

.PHONY: help install hooks format format-check lint lint-fix typecheck test check run acquire process features train train-tracked mlflow-ui mlflow-inspect mlflow-verify db-check db-upgrade db-current db-history db-downgrade db-test monitor lifecycle-status eda docker-build docker-up docker-down docker-logs docker-migrate docker-bootstrap docker-worker docker-replay-status docker-test docker-smoke

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and development dependencies
	$(UV) sync

hooks: ## Install pre-commit hooks into the local git repository
	$(UV) run pre-commit install

format: ## Format code with Ruff
	$(UV) run ruff format .

format-check: ## Check formatting without modifying files
	$(UV) run ruff format --check .

lint: ## Lint with Ruff
	$(UV) run ruff check .

lint-fix: ## Lint with Ruff and apply safe fixes
	$(UV) run ruff check --fix .

typecheck: ## Type-check src with mypy
	$(UV) run mypy src

test: ## Run the test suite
	$(UV) run pytest

check: format-check lint typecheck test ## Run all quality gates

run: ## Run the API locally with auto-reload
	$(UV) run uvicorn --factory turbine_guard.api.app:create_app --reload --host $${TURBINE_GUARD_API_HOST:-127.0.0.1} --port $${TURBINE_GUARD_API_PORT:-8000}

acquire: ## Download the NASA C-MAPSS FD001 subset into data/raw
	$(UV) run python scripts/download_data.py

process: ## Validate acquired FD001 raw files and write Parquet + report
	$(UV) run python scripts/process_data.py

features: ## Build RUL labels, asset-level splits, and features (requires make process)
	$(UV) run python scripts/build_features.py

train: ## Train and evaluate Loop 4 RUL models (requires make features)
	$(UV) run python scripts/train_models.py

train-tracked: ## Train/verify Loop 4 and track/register with MLflow
	$(UV) run python scripts/train_models.py --track-with-mlflow

mlflow-ui: ## Launch the local SQLite-backed MLflow UI
	$(UV) run mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db --port 5000

mlflow-inspect: ## Show tracked executions, candidates, registry versions, and aliases
	$(UV) run python scripts/mlflow_models.py inspect

mlflow-verify: ## Verify champion-alias predictions against the local bundle
	$(UV) run python scripts/mlflow_models.py verify --alias champion

db-check: ## Check operational PostgreSQL connectivity and migration revision
	$(UV) run alembic current

db-upgrade: ## Apply operational PostgreSQL migrations
	$(UV) run alembic upgrade head

db-current: ## Show the operational database revision
	$(UV) run alembic current

db-history: ## Show migration history
	$(UV) run alembic history

db-downgrade: ## Downgrade one revision (development only)
	$(UV) run alembic downgrade -1

db-test: ## Run guarded PostgreSQL integration tests
	$(UV) run pytest -m postgres tests/integration/test_postgres_operational.py

monitor: ## Run one Loop 9 production monitoring window
	$(UV) run python scripts/model_lifecycle.py monitor

lifecycle-status: ## Inspect monitoring/retraining/promotion pipeline runs
	$(UV) run python scripts/model_lifecycle.py status

eda: ## Execute the EDA notebook top to bottom (requires make process)
	$(UV) run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

docker-build: ## Build the reusable production application image
	docker compose build

docker-up: ## Start PostgreSQL, MLflow, migrations, and the API
	docker compose up --detach --build

docker-down: ## Stop the local stack without deleting named volumes
	docker compose --profile bootstrap --profile ops --profile replay down

docker-logs: ## Follow API, MLflow, and PostgreSQL logs
	docker compose logs --follow api mlflow postgres

docker-migrate: ## Apply Alembic migrations through the one-shot owner
	docker compose run --rm migrate

docker-bootstrap: ## Explicitly build data/models and register the initial champion
	docker compose --profile bootstrap run --rm bootstrap

docker-worker: ## Run one safe monitoring window; never auto-promotes a model
	docker compose --profile ops run --rm worker

docker-replay-status: ## Inspect replay state without starting a trajectory
	docker compose --profile replay run --rm replay

docker-test: docker-build ## Verify image user, imports, CLIs, settings, and shutdown
	sh scripts/verify_container.sh $${TURBINE_GUARD_IMAGE:-turbine-guard:local}

docker-smoke: ## Run an isolated deterministic Compose/API smoke test and clean it up
	sh scripts/docker_smoke.sh
