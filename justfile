# Task runner for the project

# Auto-detect uv - falls back to plain python if not available
PYTHON := `command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python"`

# install tooling
init:
    #!/usr/bin/env bash
    if command -v uv >/dev/null 2>&1; then
        echo "Using uv..."
        uv sync --extra dev
        uv run pre-commit install
    else
        echo "Using pip..."
        python -m pip install -U pip
        pip install -e ".[dev]"
        pre-commit install
    fi

# format code (rewrites files)
fmt:
    {{PYTHON}} -m black .

# verify formatting without rewriting (used by CI)
fmt-check:
    {{PYTHON}} -m black --check .

# lint code
lint:
    {{PYTHON}} -m ruff check .

# type-check
type:
    {{PYTHON}} -m mypy .

# run tests
test:
    {{PYTHON}} -m pytest

# run all checks, formatting the code first (local development)
all: fmt lint type test
    echo "All checks completed!"

# run all checks without modifying files (CI): fails on formatting violations
ci: fmt-check lint type test
    echo "All CI checks passed!"

# start FastAPI server (dashboard)
dashboard:
    {{PYTHON}} src/wikipedia/api/api.py

# run regression audit
audit:
    {{PYTHON}} scripts/audit_scores.py

# remove cached Wikipedia API responses (52MB+ after heavy use)
clean-cache:
    rm -rf .wiki_cache .test_wiki_cache

# remove caches and build artifacts
clean: clean-cache
    rm -rf .mypy_cache .pytest_cache .ruff_cache build dist
    find . -name "__pycache__" -type d -prune -exec rm -rf {} +




