# Wikipedia Maturity Scoring CLI

Command-line interface for scoring Wikipedia articles using the heuristic baseline model.

## Features

- Color-coded maturity level output
- Formatted summary with pillar scores
- Top contributing features display
- JSON output option for scripting
- Verbose mode for detailed information

## Installation

1. Make the script executable:
```bash
chmod +x cli/wiki_score.py
```

2. Optional: Install system-wide:
```bash
python setup_cli.py
```

## Usage

### Basic Usage

```bash
python cli/wiki_score.py "Albert Einstein"
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
python cli/wiki_score.py "Albert Einstein" --json
```

### Verbose Mode

```bash
python cli/wiki_score.py "Albert Einstein" --verbose
```

### Combined Options

```bash
python cli/wiki_score.py "Albert Einstein" --json --verbose
```

## Command Line Options

- `TITLE`: Wikipedia article title to score (required)
- `--json`: Output results in JSON format
- `--verbose, -v`: Show detailed scoring information

## Maturity Bands

- **Excellent** (80-100): High-quality, well-developed articles
- **Good** (60-79): Good quality with room for improvement
- **Fair** (40-59): Adequate but needs significant work
- **Poor** (20-39): Low quality, major issues
- **Very Poor** (0-19): Very low quality, minimal content

## Color Coding

- Green: Excellent scores
- Cyan: Good scores
- Yellow: Fair scores
- Magenta: Poor scores
- Red: Very poor scores

## Examples

```bash
# Score a famous scientist
python cli/wiki_score.py "Albert Einstein"

# Score a programming language
python cli/wiki_score.py "Python (programming language)" --json

# Get detailed information
python cli/wiki_score.py "Machine learning" --verbose

# Score multiple articles (bash)
for article in "Albert Einstein" "Python (programming language)" "Machine learning"; do
    python cli/wiki_score.py "$article" --json
done
```

## Error Handling

The CLI provides clear error messages for:
- Article not found
- Network connectivity issues
- Invalid article titles
- Scoring errors

Use `--verbose` for detailed error information including stack traces.
