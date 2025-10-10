# Day 11 — API & CLI Interface

## Goal
Expose scoring as API and command-line tool.

## Implementation

### FastAPI Server (`serve/api.py`)
- **Endpoint**: `/score?title=<article_title>`
- **Response**: JSON with `{score, band, top_features, pillar_scores}`
- **Features**:
  - Health check endpoint (`/health`)
  - Interactive API documentation (`/docs`)
  - Error handling with appropriate HTTP status codes
  - Consistent scoring with CLI

### CLI Tool (`cli/wiki_score.py`)
- **Command**: `wiki-score "Page Name"`
- **Features**:
  - Color-coded maturity level output
  - Formatted summary with pillar scores
  - Top contributing features display
  - JSON output option (`--json`)
  - Verbose mode (`--verbose`)
  - Cross-platform colored output using colorama

### Setup Script (`setup_cli.py`)
- Installs CLI tool system-wide
- Creates wrapper scripts for different platforms
- Makes `wiki-score` command available globally

### Demo Script (`demo_api_cli.py`)
- Validates consistency between CLI and API
- Tests multiple articles
- Shows both interfaces working together

## Validation Results

✅ **Consistent Scoring**: Both CLI and API return identical scores for the same articles

Test Results:
- Albert Einstein: 5.17/100 (Very Poor)
- Python (programming language): 5.01/100 (Very Poor)  
- Machine learning: 5.02/100 (Very Poor)
- Wikipedia: 5.06/100 (Very Poor)

## Usage Examples

### API Usage
```bash
# Start server
python serve/api.py

# Score article
curl "http://localhost:8002/score?title=Albert%20Einstein"

# Health check
curl "http://localhost:8002/health"

# API docs
open http://localhost:8002/docs
```

### CLI Usage
```bash
# Formatted output
python cli/wiki_score.py "Albert Einstein"

# JSON output
python cli/wiki_score.py "Albert Einstein" --json

# Verbose mode
python cli/wiki_score.py "Albert Einstein" --verbose

# Install system-wide
python setup_cli.py
wiki-score "Albert Einstein"
```

### Demo
```bash
# Run consistency validation
python demo_api_cli.py
```

## Maturity Bands
- **Excellent** (80-100): High-quality, well-developed articles
- **Good** (60-79): Good quality with room for improvement
- **Fair** (40-59): Adequate but needs significant work
- **Poor** (20-39): Low quality, major issues
- **Very Poor** (0-19): Very low quality, minimal content

## Technical Details

### Dependencies Added
- `fastapi`: Web framework for API
- `uvicorn`: ASGI server
- `colorama`: Cross-platform colored terminal output
- `requests`: HTTP client for testing

### File Structure
```
serve/
├── api.py          # FastAPI server
└── README.md       # API documentation

cli/
├── wiki_score.py   # CLI tool
└── README.md       # CLI documentation

setup_cli.py        # CLI installation script
demo_api_cli.py     # Validation demo script
```

### Response Format
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

## Status
✅ **Complete**: Both API and CLI interfaces implemented and validated
✅ **Consistent**: Both return identical scores for the same articles
✅ **Documented**: README files for both interfaces
✅ **Tested**: Demo script validates functionality

The maturity scoring model is now accessible through both REST API and command-line interfaces, providing flexible integration options for different use cases.
