"""Feature correlation analysis for Wikipedia article maturity scoring.

This module provides tools for analyzing correlations between features to identify
redundancies, multicollinearity, and feature importance. Helps optimize feature
selection and identify candidates for removal or combination.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, cast

import numpy as np


class CorrelationAnalyzer:
    """Analyze feature correlations and multicollinearity in feature datasets."""

    def __init__(self, threshold_high: float = 0.85, threshold_low: float = 0.05):
        """Initialize the correlation analyzer.

        Args:
            threshold_high: Correlation threshold for identifying high correlations (0-1).
                           Features with |correlation| > threshold_high are considered redundant.
            threshold_low: Correlation threshold for identifying low correlations (0-1).
                          Features with |correlation| < threshold_low may be uninformative.
        """
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.correlation_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.features_data: Optional[np.ndarray] = None

    def fit(self, features_list: List[Dict[str, float]]) -> None:
        """Fit the analyzer on a collection of feature dictionaries.

        Args:
            features_list: List of feature dictionaries from multiple articles.

        Raises:
            ValueError: If features_list is empty or contains no numeric data.
        """
        if not features_list:
            raise ValueError("features_list cannot be empty")

        # Extract feature names and build matrix
        all_feature_names: Set[str] = set()
        for feature_dict in features_list:
            all_feature_names.update(feature_dict.keys())

        self.feature_names = sorted(list(all_feature_names))

        if not self.feature_names:
            raise ValueError("No numeric features found in features_list")

        # Build feature matrix, handling missing values
        matrix_data = []
        for feature_dict in features_list:
            row = []
            for fname in self.feature_names:
                value = feature_dict.get(fname, np.nan)
                # Convert to float, handle non-numeric values
                try:
                    row.append(float(value) if value is not None else np.nan)
                except (ValueError, TypeError):
                    row.append(np.nan)
            matrix_data.append(row)

        self.features_data = np.array(matrix_data, dtype=float)

        # Compute correlation matrix
        # Use nanmean/nanstd to handle missing values gracefully
        self.correlation_matrix = self._compute_correlation_matrix()

    def _compute_correlation_matrix(self) -> np.ndarray:
        """Compute Pearson correlation matrix, handling NaN values.

        Returns:
            Correlation matrix of shape (n_features, n_features).
        """
        if self.features_data is None:
            raise RuntimeError("Must call fit() before computing correlation matrix")

        n_features = self.features_data.shape[1]
        corr_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                # Get valid (non-NaN) pairs
                mask = ~(
                    np.isnan(self.features_data[:, i])
                    | np.isnan(self.features_data[:, j])
                )
                valid_i = self.features_data[mask, i]
                valid_j = self.features_data[mask, j]

                if len(valid_i) > 1 and len(valid_j) > 1:
                    # Pearson correlation
                    corr = np.corrcoef(valid_i, valid_j)[0, 1]
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                else:
                    corr_matrix[i, j] = 0.0

        return cast(np.ndarray, corr_matrix)

    def get_high_correlations(
        self, exclude_self: bool = True
    ) -> List[Tuple[str, str, float]]:
        """Get pairs of features with high correlation (potential redundancy).

        Args:
            exclude_self: If True, exclude diagonal (feature with itself).

        Returns:
            List of (feature1, feature2, correlation) tuples sorted by
            correlation magnitude (highest first).
        """
        if self.correlation_matrix is None:
            raise RuntimeError("Must call fit() before getting correlations")

        high_corr_pairs = []

        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                corr = float(self.correlation_matrix[i, j])

                if abs(corr) > self.threshold_high:
                    high_corr_pairs.append(
                        (self.feature_names[i], self.feature_names[j], corr)
                    )

        # Sort by absolute correlation (highest first)
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return high_corr_pairs

    def get_low_correlations(self) -> List[Tuple[str, float]]:
        """Get features with consistently low correlation to others (weak predictors).

        A feature is considered weak if its average absolute correlation
        with other features is below threshold_low.

        Returns:
            List of (feature_name, avg_absolute_correlation) tuples sorted by
            avg correlation (lowest first).
        """
        if self.correlation_matrix is None:
            raise RuntimeError("Must call fit() before getting correlations")

        weak_features = []

        for i, fname in enumerate(self.feature_names):
            # Average correlation to other features (exclude self)
            correlations = np.delete(self.correlation_matrix[i, :], i)
            avg_abs_corr = float(np.mean(np.abs(correlations)))

            if avg_abs_corr < self.threshold_low:
                weak_features.append((fname, avg_abs_corr))

        # Sort by avg correlation (lowest first)
        weak_features.sort(key=lambda x: x[1])
        return weak_features

    def get_feature_correlation_profile(self, feature_name: str) -> Dict[str, float]:
        """Get correlation of a single feature with all others.

        Args:
            feature_name: Name of the feature to analyze.

        Returns:
            Dictionary mapping other feature names to correlation values,
            sorted by absolute correlation (highest first).

        Raises:
            ValueError: If feature_name not found in fitted features.
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in fitted features")

        if self.correlation_matrix is None:
            raise RuntimeError("Must call fit() before getting correlations")

        idx = self.feature_names.index(feature_name)
        correlations = self.correlation_matrix[idx, :].copy()

        profile = {}
        for i, fname in enumerate(self.feature_names):
            if fname != feature_name:
                profile[fname] = float(correlations[i])

        # Sort by absolute correlation
        sorted_profile = dict(
            sorted(profile.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return sorted_profile

    def get_multicollinearity_score(self) -> float:
        """Calculate overall multicollinearity score (0-1 scale).

        Uses Variance Inflation Factor (VIF) concept. Score close to 1 indicates
        high multicollinearity, close to 0 indicates low multicollinearity.

        Returns:
            Multicollinearity score (0-1 range).
        """
        if self.correlation_matrix is None:
            raise RuntimeError("Must call fit() before computing multicollinearity")

        # Use mean of pairwise absolute correlations as approximation
        n = len(self.feature_names)
        if n < 2:
            return 0.0

        # Sum all pairwise correlations (excluding diagonal)
        total_corr = np.sum(np.abs(self.correlation_matrix)) - n  # Subtract diagonal
        num_pairs = n * (n - 1) / 2

        # Average pairwise correlation
        avg_corr = total_corr / (2 * num_pairs) if num_pairs > 0 else 0.0

        # Scale to 0-1 (high correlation = high multicollinearity)
        return float(np.tanh(avg_corr))

    def suggest_features_to_remove(
        self, max_correlations: Optional[List[Tuple[str, str, float]]] = None
    ) -> List[str]:
        """Suggest features to remove based on correlation analysis.

        For each high-correlation pair, suggests removing the feature with
        lower average correlation to other features.

        Args:
            max_correlations: High correlation pairs to analyze. If None, uses
                            result from get_high_correlations().

        Returns:
            List of feature names suggested for removal, deduplicated.
        """
        if max_correlations is None:
            max_correlations = self.get_high_correlations()

        candidates_to_remove = set()

        for feat1, feat2, _ in max_correlations:
            # Get average correlation for each feature
            profile1 = self.get_feature_correlation_profile(feat1)
            profile2 = self.get_feature_correlation_profile(feat2)

            avg_corr1 = float(np.mean(np.abs(list(profile1.values()))))
            avg_corr2 = float(np.mean(np.abs(list(profile2.values()))))

            # Remove the one with lower average correlation (less informative)
            if avg_corr1 < avg_corr2:
                candidates_to_remove.add(feat1)
            else:
                candidates_to_remove.add(feat2)

        return sorted(list(candidates_to_remove))

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive correlation analysis report.

        Returns:
            Dictionary containing:
            - high_correlations: Redundant feature pairs
            - low_correlations: Weak features
            - multicollinearity_score: Overall multicollinearity metric
            - suggested_removals: Features to remove
            - summary: Human-readable summary
        """
        if self.correlation_matrix is None:
            raise RuntimeError("Must call fit() before generating report")

        high_corr = self.get_high_correlations()
        low_corr = self.get_low_correlations()
        multicollin_score = self.get_multicollinearity_score()
        removals = self.suggest_features_to_remove(high_corr)

        summary = (
            f"Analyzed {len(self.feature_names)} features across "
            f"{self.features_data.shape[0] if self.features_data is not None else 0} articles.\n"
        )

        if high_corr:
            summary += (
                f"Found {len(high_corr)} high-correlation pairs "
                f"(threshold > {self.threshold_high}).\n"
            )
        else:
            summary += "No high-correlation pairs found.\n"

        if low_corr:
            summary += (
                f"Found {len(low_corr)} weak features "
                f"(avg correlation < {self.threshold_low}).\n"
            )
        else:
            summary += "No weak features found.\n"

        if removals:
            summary += (
                f"Suggested removing {len(removals)} features "
                f"to reduce multicollinearity.\n"
            )
        else:
            summary += "No feature removals suggested.\n"

        summary += f"Multicollinearity score: {multicollin_score:.3f}"

        return {
            "high_correlations": high_corr,
            "low_correlations": low_corr,
            "multicollinearity_score": multicollin_score,
            "suggested_removals": removals,
            "summary": summary,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis results to dictionary for serialization.

        Returns:
            Dictionary containing all analysis data and results.
        """
        report = self.generate_report()

        return {
            "feature_names": self.feature_names,
            "correlation_matrix": (
                self.correlation_matrix.tolist()
                if self.correlation_matrix is not None
                else None
            ),
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
            **report,
        }
