# Wikipedia Article Maturity Scoring

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![License](https://img.shields.io/github/license/bangyen/wikipedia)](LICENSE)

**FastAPI + CLI for Wikipedia article quality assessment with ML-powered feature extraction**

[Evaluate Wikipedia article maturity using heuristic scoring and comprehensive feature analysis.]: #

## Quickstart

Clone the repo and run the demo:

```bash
git clone https://github.com/bangyen/wikipedia.git
cd wikipedia
source venv/bin/activate
pip install -e .
pytest   # optional: run tests
python api/api.py
```

Or use the CLI: `wiki-score "Albert Einstein"`

## Results

| Validation Type      | Coverage       | Result         |
|----------------------|----------------|----------------|
| Unit Tests           | 69 tests       | **Passing**    |
| Temporal Validation  | 2006-2024      | Unbiased       |
| Type Checking        | Full codebase  | mypy strict    |

## Features

- **Maturity Scoring** — Calibrated heuristic model with quality band classification.
- **FastAPI + CLI** — RESTful API with automatic docs and color-coded CLI.
- **SHAP Analysis** — Explainable AI for feature importance.

## Repo Structure

```plaintext
wikipedia/
├── examples/demo.ipynb  # Interactive demo
├── scripts/             # Validation and setup
├── tests/               # 69 unit tests
├── api/                 # FastAPI server + CLI
├── features/            # Feature extraction
├── models/              # Baseline model + weights
└── wiki_client.py       # Wikipedia API client
```

## Validation

- ✅ Full test coverage (`pytest`)
- ✅ Reproducible model weights
- ✅ Type-safe with mypy

## References

- [Wikipedia API Documentation](https://www.mediawiki.org/wiki/API:Main_page)
- [SHAP: Explainable AI](https://github.com/slundberg/shap)

## License

This project is licensed under the [MIT License](LICENSE).
