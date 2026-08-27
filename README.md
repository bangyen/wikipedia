# Wikipedia Article Maturity Scoring

[![CI](https://github.com/bangyen/wikipedia/actions/workflows/ci.yml/badge.svg)](https://github.com/bangyen/wikipedia/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/bangyen/wikipedia)](LICENSE)

**FastAPI + CLI for Wikipedia article quality assessment with ML-powered feature extraction**

[Evaluate Wikipedia article maturity using heuristic scoring and comprehensive feature analysis.]: #

## Quickstart

Clone the repo and run with [uv](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/bangyen/wikipedia.git
cd wikipedia
uv sync --all-extras     # install all dependencies (API, ML, Dev)
uv run pytest            # optional: run tests
uv run wiki-api          # start the API (or: just dashboard)
```

Or using standard pip:

```bash
git clone https://github.com/bangyen/wikipedia.git
cd wikipedia
python3.10 -m venv .venv  # Python 3.10+ required
source .venv/bin/activate
pip install -e ".[dev,api,ml]"
pytest
wiki-api                 # start the API
```

Or use the CLI: `wiki-score "Albert Einstein"`

## Results

| Validation Type      | Coverage       | Result         |
|----------------------|----------------|----------------|
| Unit Tests           | 91 tests       | **Passing**    |
| Line Coverage        | `src/`         | 73%            |
| Temporal Validation  | 2006-2024      | Unbiased       |
| Type Checking        | Full codebase  | mypy strict    |

## Features

- **Maturity Scoring** — Calibrated heuristic model with quality band classification.
- **FastAPI + CLI** — RESTful API with automatic docs and color-coded CLI.
- **SHAP Analysis** — Explainable AI for feature importance.

## Repo Structure

```plaintext
wikipedia/
├── examples/demo.ipynb       # Interactive demo
├── scripts/                  # Validation and setup
├── tests/                    # Unit and integration tests
├── src/
│   └── wikipedia/
│       ├── api/              # FastAPI server (api.py) + CLI (wiki_score.py)
│       ├── features/         # Feature extraction
│       ├── models/           # Baseline model + weights (gbm.pkl is generated)
│       └── wiki_client.py    # Wikipedia API client
└── justfile                  # Task runner
```

## Validation

- ✅ 91 tests passing (`uv run pytest`), 73% line coverage on `src/`
- ✅ Reproducible model weights (`src/wikipedia/models/weights.yaml`)
- ✅ Type-safe: `mypy` runs in `strict = true` mode across the whole repo

Coverage is uneven — feature extraction and scoring are well covered (90%+),
while the training pipeline (`models/train.py`, 21%) has smoke tests only.

## References

- [Wikipedia API Documentation](https://www.mediawiki.org/wiki/API:Main_page)
- [SHAP: Explainable AI](https://github.com/slundberg/shap)

## License

This project is licensed under the [MIT License](LICENSE).
