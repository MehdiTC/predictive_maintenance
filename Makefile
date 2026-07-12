.DEFAULT_GOAL := help

UV ?= uv

.PHONY: help install hooks format format-check lint lint-fix typecheck test check run acquire process features train train-tracked mlflow-ui mlflow-inspect mlflow-verify db-check db-upgrade db-current db-history db-downgrade db-test eda

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
	$(UV) run uvicorn --factory turbine_guard.api.app:create_app --reload

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

eda: ## Execute the EDA notebook top to bottom (requires make process)
	$(UV) run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
