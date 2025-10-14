.PHONY: init fmt lint type test dashboard all

init: ## install tooling
	python -m pip install -U pip
	pip install -e ".[dev]"
	pre-commit install

fmt:  ## format code
	black .

lint: ## lint code
	ruff check .

type: ## type-check
	mypy .

test: ## run tests
	python -m pytest

dashboard: ## run the web dashboard
	python ui/app.py

all: fmt lint type test
