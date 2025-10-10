# Heuristic Baseline Model

This directory contains the heuristic baseline model for Wikipedia article maturity scoring.

## Overview

The heuristic baseline model calculates maturity scores using a weighted average of normalized features across four pillars:

- **Structure** (25%): Article organization, formatting, and structural completeness
- **Sourcing** (30%): Citation quality, density, and source diversity  
- **Editorial** (25%): Edit history, contributor diversity, and content stability
- **Network** (20%): Connectivity within Wikipedia's knowledge network

## Files

- `baseline.py` - Main heuristic baseline model implementation
- `weights.yaml` - Configuration file containing pillar and feature weights
- `__init__.py` - Package initialization

## Usage

### Basic Usage

```python
from models.baseline import HeuristicBaselineModel

# Initialize model
model = HeuristicBaselineModel()

# Calculate maturity score for an article
result = model.calculate_maturity_score(article_data)

print(f"Maturity Score: {result['maturity_score']:.1f}/100")
print(f"Pillar Scores: {result['pillar_scores']}")
```

### Weight Calibration

```python
# Calibrate weights using training data
training_data = [(article_data, target_score), ...]
results = model.calibrate_weights(training_data, target_correlation=0.6)

print(f"Best correlation: {results['best_correlation']:.3f}")
```

### Feature Importance Analysis

```python
# Get feature importance for an article
importance = model.get_feature_importance(article_data)

# Top 5 most important features
top_features = list(importance.items())[:5]
for feature, imp in top_features:
    print(f"{feature}: {imp:.3f}")
```

## Configuration

The model uses `weights.yaml` for configuration:

```yaml
pillars:
  structure: 0.25
  sourcing: 0.30
  editorial: 0.25
  network: 0.20

features:
  section_count: 0.15
  content_length: 0.20
  citation_count: 0.25
  # ... more features
```

## Validation

The model is designed to achieve correlation ≥ 0.6 with ORES articlequality scores.

### Running Validation

```bash
# Validate correlation with ORES
python scripts/validate_model.py

# Calibrate weights using GA/FA vs Stub/Start examples
python scripts/calibrate_weights.py

# Demo the model
python scripts/demo_baseline.py
```

## Features

The model extracts features from four categories:

### Structure Features
- Section count and depth
- Content length and organization
- Template usage
- Infobox presence

### Sourcing Features
- Citation count and density
- External link analysis
- Source quality indicators
- Academic source ratio

### Editorial Features
- Editor count and diversity
- Revision history
- Recent activity
- Bot edit ratio

### Network Features
- Inbound/outbound links
- Connectivity scores
- Authority measures
- Link density

## Scoring Algorithm

1. **Feature Extraction**: Extract raw features from article data
2. **Normalization**: Normalize features to 0-1 scale
3. **Pillar Scoring**: Calculate weighted scores for each pillar
4. **Final Score**: Combine pillar scores using pillar weights
5. **Scaling**: Scale final score to 0-100 range

## Dependencies

- `numpy` - Numerical computations
- `yaml` - Configuration file parsing
- `features` - Feature extraction modules
- `src.ingest.wiki_client` - Wikipedia API client

## Testing

Run the test suite:

```bash
python -m pytest tests/unit/test_baseline.py -v
```

## Performance

The model is designed for:
- **Speed**: Fast heuristic scoring without ML training
- **Interpretability**: Clear feature weights and pillar breakdown
- **Reliability**: Robust handling of missing or incomplete data
- **Calibration**: Configurable weights for different use cases

## Future Improvements

- Machine learning-based weight optimization
- Dynamic weight adjustment based on article type
- Integration with more sophisticated feature engineering
- Real-time calibration using user feedback
