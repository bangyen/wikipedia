"""Unit tests for heuristic baseline model.

This module tests the HeuristicBaselineModel class including feature extraction,
normalization, scoring, and weight calibration functionality.
"""

import math
import numpy as np
from unittest.mock import Mock, patch, mock_open

from wikipedia.models.baseline import HeuristicBaselineModel


class TestHeuristicBaselineModel:
    """Test cases for HeuristicBaselineModel."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.model = HeuristicBaselineModel()

        # Sample article data for testing
        self.sample_article_data = {
            "title": "Test Article",
            "data": {
                "parse": {
                    "text": {"*": "This is a test article with some content."},
                    "sections": [
                        {"index": 1, "line": "Introduction", "level": 2},
                        {"index": 2, "line": "History", "level": 2},
                        {"index": 3, "line": "References", "level": 2},
                    ],
                },
                "query": {
                    "pages": {
                        "123": {
                            "extract": "This is a test article with some content.",
                            "templates": [
                                {"title": "Template:Infobox", "ns": 10},
                                {"title": "Template:Stub", "ns": 10},
                            ],
                            "revisions": [
                                {
                                    "revid": 1,
                                    "timestamp": "2024-01-01T00:00:00Z",
                                    "user": "TestUser",
                                    "comment": "Test edit",
                                    "size": 1000,
                                }
                            ],
                            "extlinks": [
                                {"url": "https://example.com"},
                                {"url": "https://academic.edu"},
                            ],
                        }
                    },
                    "sections": [
                        {"index": 1, "line": "Introduction", "level": 2},
                        {"index": 2, "line": "History", "level": 2},
                        {"index": 3, "line": "References", "level": 2},
                    ],
                    "backlinks": [{"title": "Link 1"}, {"title": "Link 2"}],
                },
            },
        }

    def test_model_initialization(self) -> None:
        """Test model initialization with default weights."""
        model = HeuristicBaselineModel()

        assert model.weights_file == "wikipedia/models/weights.yaml"
        assert "pillars" in model.weights
        assert "features" in model.weights
        # Values from weights.yaml after percentile-based calibration
        assert model.pillar_weights["structure"] == 0.2
        assert model.pillar_weights["sourcing"] == 0.35
        assert model.pillar_weights["editorial"] == 0.3
        assert model.pillar_weights["network"] == 0.15

    def test_model_initialization_with_custom_weights(self) -> None:
        """Test model initialization with custom weights file."""
        with patch("builtins.open", Mock()), patch(
            "yaml.safe_load", Mock(return_value={"pillars": {"test": 0.5}})
        ):

            model = HeuristicBaselineModel("custom_weights.yaml")
            assert model.weights_file == "custom_weights.yaml"

    def test_extract_features(self) -> None:
        """Test feature extraction from article data."""
        features = self.model.extract_features(self.sample_article_data)

        # Check that features are extracted
        assert isinstance(features, dict)
        assert len(features) > 0

        # Check specific feature types
        assert "section_count" in features
        assert "content_length" in features
        assert "citation_count" in features
        assert "total_editors" in features
        assert "inbound_links" in features

    def test_normalize_features(self) -> None:
        """Test feature normalization."""
        raw_features = {
            "section_count": 10,
            "content_length": 5000,
            "citation_count": 20,
            "has_infobox": 1.0,
            "academic_source_ratio": 0.3,
        }

        normalized = self.model.normalize_features(raw_features)

        # Check that all features are normalized to 0-1 range
        for feature, value in normalized.items():
            assert 0 <= value <= 1, f"Feature {feature} not normalized: {value}"

        # Check specific normalizations
        assert normalized["has_infobox"] == 1.0
        # academic_source_ratio: with percentile-based normalization (p10=0, p90=0.3)
        # value of 0.3 maps to (0.3 - 0) / (0.3 - 0) = 1.0
        assert 0.95 <= normalized["academic_source_ratio"] <= 1.0

    def test_calculate_pillar_scores(self) -> None:
        """Test pillar score calculation."""
        normalized_features = {
            "section_count": 0.5,
            "content_length": 0.7,
            "has_infobox": 1.0,
            "citation_count": 0.6,
            "citations_per_1k_tokens": 0.4,
            "total_editors": 0.8,
            "total_revisions": 0.9,
            "inbound_links": 0.3,
            "outbound_links": 0.5,
        }

        pillar_scores = self.model.calculate_pillar_scores(normalized_features)

        # Check that all pillars have scores
        assert "structure" in pillar_scores
        assert "sourcing" in pillar_scores
        assert "editorial" in pillar_scores
        assert "network" in pillar_scores

        # Check that scores are in 0-1 range
        for pillar, score in pillar_scores.items():
            assert 0 <= score <= 1, f"Pillar {pillar} score out of range: {score}"

    def test_calculate_maturity_score(self) -> None:
        """Test overall maturity score calculation."""
        result = self.model.calculate_maturity_score(self.sample_article_data)

        # Check result structure
        assert "maturity_score" in result
        assert "pillar_scores" in result
        assert "raw_features" in result
        assert "normalized_features" in result
        assert "weights_used" in result

        # Check score range
        assert 0 <= result["maturity_score"] <= 100

        # Check pillar scores
        for pillar, score in result["pillar_scores"].items():
            assert 0 <= score <= 100

    def test_get_feature_importance(self) -> None:
        """Test feature importance calculation."""
        importance = self.model.get_feature_importance(self.sample_article_data)

        # Check that importance is calculated
        assert isinstance(importance, dict)
        assert len(importance) > 0

        # Check that importance values are non-negative
        for feature, imp in importance.items():
            assert imp >= 0, f"Negative importance for {feature}: {imp}"

        # Check that importance is sorted
        importance_values = list(importance.values())
        assert importance_values == sorted(importance_values, reverse=True)

    def test_calibrate_weights(self) -> None:
        """Test weight calibration functionality."""
        # Create mock training data
        training_data = [
            (self.sample_article_data, 80.0),
            (self.sample_article_data, 60.0),
            (self.sample_article_data, 40.0),
        ]

        # Mock numpy.corrcoef to return a numpy array
        mock_corr_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        with patch("numpy.corrcoef", Mock(return_value=mock_corr_matrix)), patch(
            "wikipedia.models.baseline.HeuristicBaselineModel.save_weights"
        ):
            results = self.model.calibrate_weights(training_data)

        # Check calibration results
        assert "best_correlation" in results
        assert "calibrated_weights" in results
        assert "target_correlation" in results

        assert results["target_correlation"] == 0.6

    def test_save_weights(self) -> None:
        """Test weight saving functionality."""
        with patch("builtins.open", mock_open()) as mock_file, patch(
            "yaml.dump", Mock()
        ) as mock_dump:

            self.model.save_weights("test_weights.yaml")

            # Check that file was opened and yaml.dump was called
            mock_file.assert_called_once_with("test_weights.yaml", "w")
            mock_dump.assert_called_once()

    def test_empty_article_data(self) -> None:
        """Test handling of empty article data."""
        empty_data = {"title": "Empty", "data": {}}

        result = self.model.calculate_maturity_score(empty_data)

        # Should still return a valid result
        assert "maturity_score" in result
        assert 0 <= result["maturity_score"] <= 100

    def test_missing_features(self) -> None:
        """Test handling of missing features."""
        # Test with minimal data
        minimal_data = {
            "title": "Minimal",
            "data": {"query": {"pages": {"123": {"extract": "Minimal content"}}}},
        }

        result = self.model.calculate_maturity_score(minimal_data)

        # Should handle missing features gracefully
        assert "maturity_score" in result
        assert 0 <= result["maturity_score"] <= 100

    def test_weight_validation(self) -> None:
        """Test weight validation and defaults."""
        # Test that weights sum to reasonable values
        pillar_sum = sum(self.model.pillar_weights.values())
        assert (
            abs(pillar_sum - 1.0) < 0.01
        ), f"Pillar weights don't sum to 1: {pillar_sum}"

        # Test that feature weights are reasonable
        for feature, weight in self.model.feature_weights.items():
            if isinstance(weight, (int, float)):
                assert (
                    -1.0 <= weight <= 1.0
                ), f"Feature weight out of range: {feature}={weight}"

    def test_correlation_calculation(self) -> None:
        """Test correlation calculation in calibration."""
        # Mock numpy.corrcoef to return known correlation
        mock_corr_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        with patch("numpy.corrcoef", Mock(return_value=mock_corr_matrix)), patch(
            "wikipedia.models.baseline.HeuristicBaselineModel.save_weights"
        ):
            training_data = [
                (self.sample_article_data, 80.0),
                (self.sample_article_data, 60.0),
            ]

            results = self.model.calibrate_weights(training_data)

            # Check that correlation was calculated
            assert "best_correlation" in results
            assert results["best_correlation"] == 0.8

    def test_feature_extraction_edge_cases(self) -> None:
        """Test feature extraction with edge cases."""
        # Test with None values
        edge_case_data = {
            "title": "Edge Case",
            "data": {
                "parse": {"text": {"*": ""}},
                "query": {
                    "pages": {
                        "123": {
                            "extract": "",
                            "templates": [],
                            "revisions": [],
                            "extlinks": [],
                        }
                    },
                    "sections": [],
                    "backlinks": [],
                },
            },
        }

        features = self.model.extract_features(edge_case_data)

        # Should handle edge cases gracefully
        assert isinstance(features, dict)
        assert len(features) > 0

        # Check that numeric features are valid
        for feature, value in features.items():
            assert isinstance(value, (int, float))
            assert not math.isnan(value)
            assert not math.isinf(value)
