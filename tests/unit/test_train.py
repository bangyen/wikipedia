"""Smoke tests for the training pipeline.

These cover construction, error paths, and save/load round-tripping. They do
not exercise real LightGBM training, which requires a full dataset — meaningful
coverage of the training loop itself remains open.
"""

from pathlib import Path
from typing import Any, Dict

import pytest

lightgbm = pytest.importorskip("lightgbm", reason="training extras not installed")

from wikipedia.models.train import WikipediaMaturityClassifier  # noqa: E402


def test_classifier_initializes_with_reproducible_seed() -> None:
    """Classifier records its seed and stays untrained until fit."""
    clf = WikipediaMaturityClassifier(random_state=7)

    assert clf.random_state == 7
    assert clf.lgb_params["random_state"] == 7
    assert clf.model is None
    assert clf.feature_names is None


def test_lgb_params_configured_for_binary_classification() -> None:
    """Default params target binary classification deterministically."""
    params = WikipediaMaturityClassifier().lgb_params

    assert params["objective"] == "binary"
    assert params["metric"] == "binary_logloss"
    assert params["random_state"] == 42


def test_save_model_without_training_raises() -> None:
    """Saving before training is an explicit error, not a silent no-op."""
    clf = WikipediaMaturityClassifier()

    with pytest.raises(ValueError, match="No model to save"):
        clf.save_model("unused.pkl")


def test_save_load_round_trip_preserves_state(tmp_path: Path) -> None:
    """A saved model reloads with its params and feature names intact."""
    clf = WikipediaMaturityClassifier(random_state=13)
    # Stand in for a fitted booster; save/load only pickles what it is given.
    clf.model = {"stub": "booster"}
    clf.feature_names = ["citation_count", "section_count"]

    path = tmp_path / "model.pkl"
    clf.save_model(str(path))
    assert path.exists()

    reloaded = WikipediaMaturityClassifier(random_state=999)
    reloaded.load_model(str(path))

    assert reloaded.model == {"stub": "booster"}
    assert reloaded.feature_names == ["citation_count", "section_count"]
    assert reloaded.random_state == 13
    assert reloaded.lgb_params["objective"] == "binary"


def test_get_feature_importance_without_model_raises() -> None:
    """Feature importance is unavailable before training."""
    clf = WikipediaMaturityClassifier()

    with pytest.raises((ValueError, AttributeError)):
        clf.get_feature_importance()


def test_extract_all_features_returns_numeric_mapping() -> None:
    """Feature extraction yields a flat numeric dict for a minimal article."""
    clf = WikipediaMaturityClassifier()
    article_data: Dict[str, Any] = {"title": "Test", "data": {}}

    features = clf.extract_all_features(article_data)

    assert isinstance(features, dict)
    assert features, "expected at least one extracted feature"
    assert all(isinstance(v, (int, float)) for v in features.values())
