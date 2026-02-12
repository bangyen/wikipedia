# Feature Correlation Analysis

This module provides tools to analyze feature correlations and identify optimization opportunities in your Wikipedia maturity scoring system.

## Overview

The `CorrelationAnalyzer` class helps you:

1. **Detect Redundant Features** - Identify highly correlated feature pairs (multicollinearity)
2. **Find Weak Features** - Identify features with low predictive power
3. **Assess Overall Multicollinearity** - Measure the overall correlation structure
4. **Suggest Optimizations** - Get recommendations for feature removal

## Quick Start

```python
from features.correlation_analysis import CorrelationAnalyzer
from features.extractors import structure_features, sourcing_features, editorial_features, network_features

# Collect features from multiple articles
features_list = []
for article in articles:
    features = {}
    features.update(structure_features(article))
    features.update(sourcing_features(article))
    features.update(editorial_features(article))
    features.update(network_features(article))
    features_list.append(features)

# Analyze correlations
analyzer = CorrelationAnalyzer(threshold_high=0.85, threshold_low=0.05)
analyzer.fit(features_list)

# Get results
report = analyzer.generate_report()
print(report["summary"])
```

## Key Concepts

### Correlation Types

**High Correlation (redundancy)**
- When two features have |correlation| > `threshold_high` (default: 0.85)
- Indicates features measure similar information
- Suggests removing one of the pair

Example: `citation_count` and `citation_density` might be highly correlated because longer articles naturally have more citations.

**Low Correlation (weak predictor)**
- When a feature has avg |correlation| < `threshold_low` (default: 0.05)
- Indicates feature is independent from others
- May indicate weak predictive power

Example: A feature that fluctuates randomly across articles.

### Multicollinearity Score

A 0-1 score indicating overall feature redundancy:
- **< 0.3**: Low multicollinearity (good)
- **0.3-0.6**: Moderate multicollinearity
- **> 0.6**: High multicollinearity (problematic)

High multicollinearity can cause:
- Unstable weight calibration
- Difficulty interpreting feature importance
- Overfitting to training data

## API Reference

### CorrelationAnalyzer Class

#### Initialization
```python
analyzer = CorrelationAnalyzer(threshold_high=0.85, threshold_low=0.05)
```

**Parameters:**
- `threshold_high` (float): Correlation threshold for redundancy (0-1)
- `threshold_low` (float): Correlation threshold for weak features (0-1)

#### Methods

**fit(features_list)**
Fit the analyzer on feature data.

```python
analyzer.fit([
    {"feature_a": 1.0, "feature_b": 2.0},
    {"feature_a": 2.0, "feature_b": 4.0},
    ...
])
```

**get_high_correlations(exclude_self=True)**
Get feature pairs with high correlation (potential redundancy).

Returns: `List[Tuple[str, str, float]]` - List of (feature1, feature2, correlation)

```python
high_corr = analyzer.get_high_correlations()
for feat1, feat2, corr in high_corr:
    print(f"{feat1} <-> {feat2}: r={corr:.3f}")
```

**get_low_correlations()**
Get features with consistently low correlation to others.

Returns: `List[Tuple[str, float]]` - List of (feature_name, avg_abs_correlation)

```python
weak_features = analyzer.get_low_correlations()
for feat, avg_corr in weak_features:
    print(f"{feat}: avg |r|={avg_corr:.3f}")
```

**get_feature_correlation_profile(feature_name)**
Get correlation of one feature with all others.

Returns: `Dict[str, float]` - Mapping of other features to correlation values

```python
profile = analyzer.get_feature_correlation_profile("citation_count")
for other_feat, corr in profile.items():
    print(f"{other_feat}: {corr:+.3f}")
```

**get_multicollinearity_score()**
Calculate overall multicollinearity score (0-1).

Returns: `float` - Multicollinearity score

```python
score = analyzer.get_multicollinearity_score()
if score > 0.6:
    print("⚠️  High multicollinearity detected")
```

**suggest_features_to_remove(max_correlations=None)**
Suggest features to remove based on correlation analysis.

Returns: `List[str]` - Feature names recommended for removal

```python
removals = analyzer.suggest_features_to_remove()
print(f"Consider removing: {removals}")
```

**generate_report()**
Generate comprehensive analysis report.

Returns: `Dict[str, Any]` - Complete analysis results

```python
report = analyzer.generate_report()
print(report["summary"])
print(f"High correlations: {len(report['high_correlations'])}")
print(f"Weak features: {len(report['low_correlations'])}")
print(f"Suggested removals: {report['suggested_removals']}")
```

**to_dict()**
Convert results to dictionary for serialization.

Returns: `Dict[str, Any]` - Serializable analysis results

```python
results = analyzer.to_dict()
json.dump(results, open("analysis.json", "w"))
```

## Usage Examples

### Example 1: Analyze Your Feature Set

```python
from features.correlation_analysis import CorrelationAnalyzer
from models.baseline import HeuristicBaselineModel
import json

# Collect features from 50 articles
model = HeuristicBaselineModel()
features_list = []
for article in articles:
    features = model.extract_features(article)
    features_list.append(features)

# Analyze
analyzer = CorrelationAnalyzer()
analyzer.fit(features_list)
report = analyzer.generate_report()

# View results
print(report["summary"])
```

### Example 2: Remove Redundant Features

```python
# Identify redundant features
high_corr = analyzer.get_high_correlations()

# Remove suggested features from model
removals = analyzer.suggest_features_to_remove(high_corr)
print(f"Removing: {removals}")

# Update feature extraction by filtering in your model
filtered_features = {k: v for k, v in features.items() if k not in removals}
```

### Example 3: Optimize Feature Weights

```python
# High multicollinearity can destabilize weight calibration
score = analyzer.get_multicollinearity_score()

if score > 0.6:
    print("Consider removing correlated features before calibrating weights")
    removals = analyzer.suggest_features_to_remove()
    # Remove features from training process
```

### Example 4: Identify Low-Signal Features

```python
# Find features that don't correlate well with others
weak_features = analyzer.get_low_correlations()

for feat, avg_corr in weak_features:
    print(f"⚠️  {feat} has weak signal (avg |r|={avg_corr:.3f})")
    print(f"   Consider removing or engineering a better version")
```

## Running Analysis Scripts

### Demo Script
```bash
uv run scripts/demo_correlation_analysis.py
```

Runs 5 interactive demonstrations showing:
1. Basic correlation computation
2. Redundancy detection
3. Weak feature detection
4. Multicollinearity assessment
5. Full analysis report

### Full Analysis Script
```bash
uv run scripts/analyze_correlations.py --sample-size 50
```

Fetches real Wikipedia articles and analyzes their features.

**Options:**
- `--sample-size`: Number of articles to analyze (default: 50)
- `--threshold-high`: Redundancy threshold (default: 0.85)
- `--threshold-low`: Weak feature threshold (default: 0.05)
- `--output`: Output file path (default: output/correlation_report.json)

Example:
```bash
uv run scripts/analyze_correlations.py \
    --sample-size 100 \
    --threshold-high 0.80 \
    --output results/analysis.json
```

## Understanding the Results

### High Correlation Report
```
🔴 HIGH CORRELATION PAIRS (Potential Redundancy):
  citation_count <-> citation_density | r=+0.923
  external_link_count <-> outbound_links | r=+0.910
```

**Interpretation**:
- These feature pairs measure similar information
- Consider removing one from each pair
- Reduces multicollinearity and improves weight stability

### Low Correlation Report
```
🟡 WEAK FEATURES (Low Correlation to Others):
  has_disambiguation | avg |r|=0.012
  has_see_also | avg |r|=0.028
```

**Interpretation**:
- These features don't correlate with other features
- Might be weak predictors of article quality
- Consider removing unless they have strong independent importance

### Multicollinearity Score
```
Multicollinearity score: 0.742 ❌ HIGH (problematic)
```

**Interpretation**:
- Score > 0.6 indicates significant redundancy
- Can cause instability in weight calibration
- Consider removing redundant features
- Feature importance estimates may be unreliable

## Tips for Interpretation

1. **Beware of spurious correlations**
   - Strong correlations don't imply causation
   - Both features might depend on article length

2. **Consider domain knowledge**
   - Some correlations are expected
   - `citation_count` and `content_length` naturally correlate
   - But that's why we measure both the relationship

3. **Balance precision and interpretability**
   - Removing features improves model stability
   - But you lose interpretability of each pillar
   - Find the sweet spot for your use case

4. **Validate with domain experts**
   - Correlation analysis is data-driven
   - Verify recommendations with Wikipedia editors
   - Some correlated features might both be important

## Common Patterns

### Pattern 1: Count Features Correlate
```
citation_count <-> section_count: +0.92
citation_count <-> external_links: +0.89
```
**Solution**: Keep only most important count features

### Pattern 2: Per-1k Features Redundant with Counts
```
citations_per_1k_tokens <-> citation_count: +0.85
```
**Solution**: Choose either raw count or normalized version

### Pattern 3: All Features Correlate (Healthy)
Average |correlation| = 0.6-0.7
**Meaning**: Features measure related but distinct aspects of quality

### Pattern 4: Some Features Isolated
```
has_stub: avg |r| = 0.02
```
**Meaning**: Binary features often isolated; may still be useful

## Advanced Usage

### Custom Thresholds
```python
# More strict redundancy detection
strict = CorrelationAnalyzer(threshold_high=0.90, threshold_low=0.1)
strict.fit(features_list)
```

### Correlation Matrix Access
```python
# Direct access to correlation matrix
corr_matrix = analyzer.correlation_matrix
print(corr_matrix.shape)  # (n_features, n_features)
```

### Feature Importance by Correlation
```python
# Features that correlate well with others are more "general"
for feat in analyzer.feature_names:
    profile = analyzer.get_feature_correlation_profile(feat)
    avg_corr = np.mean(np.abs(list(profile.values())))
    print(f"{feat}: {avg_corr:.3f}")  # Higher = more general
```

## Integration with Model

### Before Weight Calibration
```python
# Check multicollinearity before calibrating
analyzer = CorrelationAnalyzer()
analyzer.fit(training_features)
score = analyzer.get_multicollinearity_score()

if score > 0.6:
    removals = analyzer.suggest_features_to_remove()
    print(f"Consider removing: {removals}")
```

### Feature Selection Pipeline
```python
# 1. Extract features
features_list = [model.extract_features(article) for article in articles]

# 2. Analyze correlations
analyzer = CorrelationAnalyzer()
analyzer.fit(features_list)

# 3. Get recommendations
removals = analyzer.suggest_features_to_remove()

# 4. Remove and re-calibrate
model.feature_weights = {k: v for k, v in model.feature_weights.items()
                         if k not in removals}
model.calibrate_weights(training_data)
```

## Testing

Run unit tests:
```bash
uv run pytest tests/unit/test_correlation_analysis.py -v
```

Tests cover:
- Correlation matrix computation
- High/low correlation detection
- Multicollinearity scoring
- Feature removal suggestions
- Missing value handling
- Report generation

## See Also

- **Model Calibration**: `models/baseline.py` - Weight calibration that benefits from low multicollinearity
- **Feature Extractors**: `features/extractors.py` - Feature extraction functions
- **Baseline Model**: `models/baseline.py` - Main scoring model
