# Scripts

Operational scripts for validation, calibration, and analysis. Run them with
`uv run python scripts/<name>.py`.

## Which script when

| Script | Use it when |
|---|---|
| `audit_scores.py` | Quick regression check that scores haven't drifted. Wired to `just audit`. |
| `validate_model.py` | General model validation against labelled articles (`ModelValidator`). |
| `temporal_validation.py` | Checking for **age bias** specifically — new (<90d) vs. old articles (`TemporalValidator`). |
| `recalibrate_baseline.py` | Standard weight recalibration (`BaselineRecalibrator`, class-based). |
| `aggressive_recalibration.py` | Standard recalibration was insufficient — applies temporal-aware and age-normalized weight strategies. |
| `analyze_correlations.py` | Finding redundant/weak features before calibrating weights. |
| `shap_analysis.py` | Explaining feature importance for a trained `gbm.pkl`. |
| `generate_features.py` | Building a feature table (`features.parquet`) from articles. |
| `bulk_process_wikipedia.py` | Bulk-scoring many articles. |
| `generate_research_data.py` | Producing consolidated data for the portfolio site. |
| `validate_wikidata.py` | Sanity-checking Wikidata-derived features. |
| `setup_cli.py` | One-time CLI setup helper. |

`validate_model.py` and `temporal_validation.py` overlap in purpose but expose
different classes and answer different questions; likewise the two recalibration
scripts. They are kept separate deliberately — `generate_research_data.py`
imports `TemporalValidator` directly.

## Caching

Most scripts hit the Wikipedia API and cache responses in `.wiki_cache/`, which
grows large (50MB+). Clear it with `just clean-cache`.
