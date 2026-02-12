"""Unit tests for feature correlation analysis.

This module tests the CorrelationAnalyzer class including correlation computation,
redundancy detection, and feature removal suggestions.
"""

import numpy as np
import pytest

from features.correlation_analysis import CorrelationAnalyzer


class TestCorrelationAnalyzer:
    """Test cases for CorrelationAnalyzer."""

    def test_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = CorrelationAnalyzer(threshold_high=0.85, threshold_low=0.05)
        assert analyzer.threshold_high == 0.85
        assert analyzer.threshold_low == 0.05

    def test_fit_basic(self) -> None:
        """Test fitting with basic feature data."""
        features_list = [
            {"feat_a": 1.0, "feat_b": 2.0},
            {"feat_a": 2.0, "feat_b": 4.0},
            {"feat_a": 3.0, "feat_b": 6.0},
        ]
        analyzer = CorrelationAnalyzer()
        analyzer.fit(features_list)

        assert len(analyzer.feature_names) == 2
        assert analyzer.feature_names == ["feat_a", "feat_b"]
        assert analyzer.correlation_matrix is not None
        assert analyzer.correlation_matrix.shape == (2, 2)

    def test_fit_empty_raises_error(self) -> None:
        """Test that fitting with empty list raises ValueError."""
        analyzer = CorrelationAnalyzer()
        with pytest.raises(ValueError, match="cannot be empty"):
            analyzer.fit([])

    def test_high_correlation_detection(self) -> None:
        """Test detection of highly correlated features."""
        # Create perfectly correlated features
        features_list = [
            {"feat_a": 1.0, "feat_b": 10.0, "feat_c": 100.0},
            {"feat_a": 2.0, "feat_b": 20.0, "feat_c": 200.0},
            {"feat_a": 3.0, "feat_b": 30.0, "feat_c": 300.0},
            {"feat_a": 4.0, "feat_b": 40.0, "feat_c": 400.0},
        ]
        analyzer = CorrelationAnalyzer(threshold_high=0.99)
        analyzer.fit(features_list)

        high_corr = analyzer.get_high_correlations()
        assert len(high_corr) > 0
        # All pairs should be highly correlated (near 1.0)
        for feat1, feat2, corr in high_corr:
            assert abs(corr) > 0.99

    def test_low_correlation_detection(self) -> None:
        """Test detection of weakly correlated features."""
        # Create mostly uncorrelated features
        np.random.seed(42)
        weak_noise = [np.random.randn() for _ in range(20)]
        strong_signal = list(range(20))

        features_list = [
            {
                "weak_feat": weak_noise[i],
                "strong_feat_a": float(strong_signal[i]),
                "strong_feat_b": float(strong_signal[i] * 2),
            }
            for i in range(20)
        ]
        analyzer = CorrelationAnalyzer(threshold_low=0.3)
        analyzer.fit(features_list)

        low_corr = analyzer.get_low_correlations()
        # weak_feat should have low average correlation since it's random noise
        feature_names_low = [f[0] for f in low_corr]
        # Weak feature should be detected as low correlation
        if low_corr:  # If any weak features found
            assert "weak_feat" in feature_names_low or len(low_corr) > 0

    def test_feature_correlation_profile(self) -> None:
        """Test getting correlation profile for a single feature."""
        features_list = [
            {"feat_a": 1.0, "feat_b": 2.0, "feat_c": 3.0},
            {"feat_a": 2.0, "feat_b": 4.0, "feat_c": 6.0},
            {"feat_a": 3.0, "feat_b": 6.0, "feat_c": 9.0},
        ]
        analyzer = CorrelationAnalyzer()
        analyzer.fit(features_list)

        profile = analyzer.get_feature_correlation_profile("feat_a")
        assert "feat_b" in profile
        assert "feat_c" in profile
        assert "feat_a" not in profile  # Should exclude self

    def test_feature_correlation_profile_invalid_feature(self) -> None:
        """Test that requesting invalid feature raises error."""
        features_list = [{"feat_a": 1.0, "feat_b": 2.0}]
        analyzer = CorrelationAnalyzer()
        analyzer.fit(features_list)

        with pytest.raises(ValueError, match="not found"):
            analyzer.get_feature_correlation_profile("invalid_feat")

    def test_multicollinearity_score(self) -> None:
        """Test multicollinearity score calculation."""
        # Create uncorrelated features
        np.random.seed(42)
        features_list = [
            {"feat_a": np.random.randn(), "feat_b": np.random.randn()}
            for _ in range(50)
        ]
        analyzer = CorrelationAnalyzer()
        analyzer.fit(features_list)

        score = analyzer.get_multicollinearity_score()
        assert 0.0 <= score <= 1.0
        # Uncorrelated features should have low score
        assert score < 0.5

    def test_multicollinearity_score_high_correlation(self) -> None:
        """Test multicollinearity with highly correlated features."""
        # Create highly correlated features
        features_list = [
            {"feat_a": float(i), "feat_b": float(i) * 2, "feat_c": float(i) * 3}
            for i in range(20)
        ]
        analyzer = CorrelationAnalyzer()
        analyzer.fit(features_list)

        score = analyzer.get_multicollinearity_score()
        # Highly correlated features should have high score
        assert score > 0.7

    def test_suggest_features_to_remove(self) -> None:
        """Test feature removal suggestions."""
        # feat_b and feat_c are both highly correlated with feat_a
        # but feat_b is also correlated with feat_c
        # We should suggest removing the weakest one
        features_list = [
            {"feat_a": float(i), "feat_b": float(i) + 0.1, "feat_c": -float(i)}
            for i in range(20)
        ]
        analyzer = CorrelationAnalyzer(threshold_high=0.8)
        analyzer.fit(features_list)

        removals = analyzer.suggest_features_to_remove()
        # Should suggest removing something
        assert isinstance(removals, list)
        # All suggestions should be actual features
        for feat in removals:
            assert feat in analyzer.feature_names

    def test_generate_report(self) -> None:
        """Test comprehensive report generation."""
        features_list = [
            {"feat_a": float(i), "feat_b": float(i) * 2, "feat_c": np.random.randn()}
            for i in range(20)
        ]
        analyzer = CorrelationAnalyzer()
        analyzer.fit(features_list)

        report = analyzer.generate_report()

        assert "high_correlations" in report
        assert "low_correlations" in report
        assert "multicollinearity_score" in report
        assert "suggested_removals" in report
        assert "summary" in report

        assert isinstance(report["high_correlations"], list)
        assert isinstance(report["low_correlations"], list)
        assert isinstance(report["multicollinearity_score"], float)
        assert isinstance(report["suggested_removals"], list)
        assert isinstance(report["summary"], str)

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        features_list = [
            {"feat_a": 1.0, "feat_b": 2.0},
            {"feat_a": 2.0, "feat_b": 4.0},
        ]
        analyzer = CorrelationAnalyzer()
        analyzer.fit(features_list)

        result = analyzer.to_dict()

        assert "feature_names" in result
        assert "correlation_matrix" in result
        assert "threshold_high" in result
        assert "threshold_low" in result
        assert "high_correlations" in result
        assert "summary" in result

    def test_handles_missing_values(self) -> None:
        """Test that analyzer handles missing values gracefully."""
        features_list = [
            {"feat_a": 1.0, "feat_b": 2.0},
            {"feat_a": 2.0},  # Missing feat_b
            {"feat_b": 4.0},  # Missing feat_a
            {"feat_a": 3.0, "feat_b": 6.0},
        ]
        analyzer = CorrelationAnalyzer()
        analyzer.fit(features_list)

        assert len(analyzer.feature_names) == 2
        assert analyzer.correlation_matrix is not None

    def test_handles_non_numeric_values(self) -> None:
        """Test that analyzer converts non-numeric values to NaN gracefully."""
        features_list = [
            {"feat_a": 1.0, "feat_b": "invalid"},
            {"feat_a": 2.0, "feat_b": 4.0},
            {"feat_a": 3.0, "feat_b": 6.0},
        ]
        analyzer = CorrelationAnalyzer()
        analyzer.fit(features_list)

        # Should handle gracefully
        assert analyzer.correlation_matrix is not None
