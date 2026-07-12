.DEFAULT_GOAL := help

UV ?= uv

.PHONY: help install hooks format format-check lint lint-fix typecheck test check run acquire

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
