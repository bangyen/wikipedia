# Wikipedia Maturity Scoring API

FastAPI server for scoring Wikipedia articles using the heuristic baseline model.

## Features

- REST API endpoint `/score` for article maturity scoring
- JSON response with score, band, and top contributing features
- Health check endpoint
- Interactive API documentation at `/docs`

## Installation

1. Install dependencies:
```bash
pip install fastapi uvicorn
```

2. Start the server:
```bash
python serve/api.py
```

The API will be available at `http://localhost:8002`

## Usage

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

## Maturity Bands

- **Excellent** (80-100): High-quality, well-developed articles
- **Good** (60-79): Good quality with room for improvement
- **Fair** (40-59): Adequate but needs significant work
- **Poor** (20-39): Low quality, major issues
- **Very Poor** (0-19): Very low quality, minimal content

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `404`: Article not found
- `500`: Internal server error
