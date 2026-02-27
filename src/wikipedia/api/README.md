# Wikipedia Maturity Scoring API & CLI

FastAPI server and command-line tool for scoring Wikipedia articles using the heuristic baseline model.

## Features

- **REST API** — `/score` endpoint for article maturity scoring
- **CLI Tool** — Color-coded terminal interface with JSON export
- **Health Check** — `/health` endpoint for monitoring
- **Interactive Docs** — Auto-generated API documentation at `/docs`

## Installation

Install dependencies:
```bash
pip install -e .
```

## API Usage

### Start the Server

```bash
python api/api.py
```

The API will be available at `http://localhost:8002`

### Score an Article

```bash
curl "http://localhost:8002/score?title=Albert%20Einstein"
```

Response:
```json
{
  "title": "Albert Einstein",
  "score": 5.17,
  "band": "Very Poor",
  "top_features": {
    "authority_score": 0.2,
    "link_density": 0.15,
    "connectivity_score": 0.12,
    "token_count": 0.063,
    "total_links": 0.039
  },
  "pillar_scores": {
    "structure": 0.73,
    "sourcing": 0.0,
    "editorial": 0.0,
    "network": 49.5
  }
}
```

### Health Check

```bash
curl "http://localhost:8002/health"
```

### API Documentation

Visit `http://localhost:8002/docs` for interactive API documentation.

## CLI Usage

### Setup (One-Time)

```bash
python scripts/setup_cli.py
```

This installs the `wiki-score` command globally.

### Score an Article

```bash
wiki-score "Albert Einstein"
```

Output:
```
============================================================
Wikipedia Article Maturity Score
============================================================

Article: Albert Einstein
Score: 5.2/100 (Very Poor)

Pillar Scores:
  Structure   : 0.7/100
  Sourcing    : 0.0/100
  Editorial   : 0.0/100
  Network     : 49.5/100

Top Contributing Features:
  1. authority_score: 0.200
  2. link_density: 0.150
  3. connectivity_score: 0.120
  4. token_count: 0.063
  5. total_links: 0.039

============================================================
```

### JSON Output

```bash
wiki-score "Albert Einstein" --json
```

### Verbose Mode

```bash
wiki-score "Albert Einstein" --verbose
```

## Maturity Bands

| Band       | Score Range | Description                        |
|------------|-------------|------------------------------------|
| Excellent  | 80-100      | High-quality, well-developed       |
| Good       | 60-79       | Good quality, some improvements    |
| Fair       | 40-59       | Adequate, needs significant work   |
| Poor       | 20-39       | Low quality, major issues          |
| Very Poor  | 0-19        | Very low quality, minimal content  |

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `404`: Article not found
- `500`: Internal server error

The CLI exits with non-zero status codes on errors and supports `--verbose` for debugging.
