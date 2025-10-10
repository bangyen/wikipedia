# Wikipedia Article Maturity Scoring

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![License](https://img.shields.io/github/license/bangyen/wikipedia)](LICENSE)

**AI-powered Wikipedia article maturity assessment: FastAPI + CLI with comprehensive feature extraction**

A machine learning system that evaluates Wikipedia article maturity using heuristic scoring, feature extraction, and provides both REST API and command-line interfaces for easy integration.

## Quickstart

### API Server
```bash
# Start the FastAPI server
python api/api.py
# API available at http://localhost:8002
# Documentation at http://localhost:8002/docs
```

### CLI Tool
```bash
# Setup CLI (one-time)
python scripts/setup_cli.py

# Score an article
wiki-score "Albert Einstein"
wiki-score "Albert Einstein" --json
```

### Development Setup
```bash
git clone https://github.com/bangyen/wikipedia.git
cd wikipedia
source venv/bin/activate
pip install -e .
pytest  # Run tests
```

## Features

- **🎯 Maturity Scoring** — Heuristic baseline model with calibrated weights
- **🔗 Feature Extraction** — Link graph analysis, Wikidata integration, content metrics
- **🚀 FastAPI Server** — RESTful API with automatic documentation
- **💻 CLI Interface** — Command-line tool with color-coded output
- **📊 SHAP Analysis** — Explainable AI for feature importance
- **⚡ Type Safety** — Full mypy type checking and validation

## Repository Structure

```plaintext
wikipedia/
├── api/                    # API and CLI interfaces
│   ├── api.py             # FastAPI server
│   ├── wiki_score.py      # CLI tool
│   └── README.md          # API documentation
├── features/               # Feature extraction modules
│   ├── extractors.py      # Core feature extractors
│   ├── linkgraph.py       # Network analysis
│   └── wikidata.py        # Wikidata integration
├── models/                 # ML models and weights
│   ├── baseline.py        # Heuristic baseline model
│   ├── train.py           # LightGBM training
│   └── weights.yaml       # Calibrated model weights
├── scripts/                # Utility scripts
│   ├── generate_features.py
│   ├── validate_model.py
│   └── setup_cli.py
├── tests/                  # Test suite
├── examples/               # Demo scripts and notebooks
├── output/                 # Analysis reports and results
├── ui/                     # Web dashboard
└── wiki_client.py          # Wikipedia API client
```

## API Endpoints

- `GET /score?title=<article>` — Calculate maturity score
- `GET /health` — Health check
- `GET /docs` — Interactive API documentation

## Maturity Bands

- **Stub** (0-20): Basic article structure
- **Start** (20-40): Developing content
- **C-Class** (40-60): Good quality
- **B-Class** (60-80): High quality
- **A-Class** (80-90): Excellent quality
- **GA** (90-95): Good article
- **FA** (95-100): Featured article

## Validation

- ✅ 69 unit tests passing
- ✅ Full type checking with mypy
- ✅ Code formatting with Black and Ruff
- ✅ Pre-commit hooks for quality assurance
- ✅ Temporal validation and bias analysis
- ✅ SHAP explainability analysis

## Development History

See `examples/` for demo scripts and notebooks:
- Core feature extraction and baseline model
- Model training and validation
- API and CLI interfaces

## License

This project is licensed under the [MIT License](LICENSE).
