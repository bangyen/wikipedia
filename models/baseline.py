"""Heuristic baseline model for Wikipedia article maturity scoring.

This module implements a weighted average approach to calculate maturity scores
based on normalized features from structural, sourcing, editorial, and network
pillars. The weights are configurable via YAML and can be calibrated using
GA/FA vs Stub/Start article examples.

Features are normalized using percentile-based ranges derived from training data,
enabling adaptive normalization that reflects actual feature distributions.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml  # type: ignore
from scipy.optimize import minimize  # type: ignore

from features import (
    structure_features,
    sourcing_features,
    editorial_features,
    network_features,
)


class HeuristicBaselineModel:
    """Heuristic baseline model for Wikipedia article maturity scoring.

    This model calculates maturity scores using a weighted average of normalized
    features across four pillars: structure, sourcing, editorial, and network.
    The weights are configurable and can be calibrated against known quality
    indicators like ORES articlequality scores.

    Normalization uses percentile-based ranges that adapt to the distribution of
    feature values in the training data, avoiding hard-coded thresholds.
    """

    def __init__(
        self,
        weights_file: Optional[str] = None,
        normalization_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """Initialize the heuristic baseline model.

        Args:
            weights_file: Path to YAML file containing feature weights.
                         If None, uses default weights.
            normalization_ranges: Dictionary mapping feature names to (percentile_low, percentile_high)
                                 tuples representing the normalization range. If None, uses defaults.
        """
        self.weights_file = weights_file or "models/weights.yaml"
        self.normalization_ranges: Dict[str, Tuple[float, float]] = {}
        self.weights = self._load_weights()
        self.pillar_weights = self.weights.get("pillars", {})
        self.feature_weights = self.weights.get("features", {})

        # Use provided ranges, or load from weights file, or use defaults
        if normalization_ranges:
            self.normalization_ranges = normalization_ranges

        # If still empty after loading weights, use defaults
        if not self.normalization_ranges:
            self.normalization_ranges = self._get_default_normalization_ranges()

    def _load_weights(self) -> Dict[str, Any]:
        """Load weights and normalization ranges from YAML configuration file.

        Returns:
            Dictionary containing pillar, feature weights, and normalization ranges.

        Raises:
            FileNotFoundError: If weights file doesn't exist.
            yaml.YAMLError: If YAML file is malformed.
        """
        weights_path = Path(self.weights_file)

        if not weights_path.exists():
            # Return default weights if file doesn't exist
            return self._get_default_weights()

        try:
            with open(weights_path, "r") as f:
                weights = yaml.safe_load(f)

            # Convert normalization_ranges back to tuple format if present
            if weights and "normalization_ranges" in weights:
                norm_ranges = weights.get("normalization_ranges", {})
                if isinstance(norm_ranges, dict):
                    self.normalization_ranges = {
                        k: tuple(v) if isinstance(v, list) else v
                        for k, v in norm_ranges.items()
                    }

            return weights  # type: ignore
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing weights file {self.weights_file}: {e}")

    def _get_default_weights(self) -> Dict[str, Any]:
        """Get default weights for the heuristic model.

        Returns:
            Dictionary containing default pillar and feature weights.
        """
        return {
            "pillars": {
                "structure": 0.25,
                "sourcing": 0.30,
                "editorial": 0.25,
                "network": 0.20,
            },
            "features": {
                # Structure features
                "section_count": 0.15,
                "content_length": 0.20,
                "has_infobox": 0.10,
                "template_count": 0.10,
                "avg_section_depth": 0.10,
                "sections_per_1k_chars": 0.10,
                "has_references": 0.10,
                "has_external_links": 0.15,
                # Sourcing features
                "citation_count": 0.25,
                "citations_per_1k_tokens": 0.20,
                "external_link_count": 0.15,
                "citation_density": 0.15,
                "has_reliable_sources": 0.15,
                "academic_source_ratio": 0.10,
                # Editorial features
                "total_editors": 0.20,
                "total_revisions": 0.20,
                "editor_diversity": 0.15,
                "recent_activity_score": 0.15,
                "bot_edit_ratio": -0.10,  # Negative weight for bot edits
                "major_editor_ratio": 0.20,
                # Network features
                "inbound_links": 0.25,
                "outbound_links": 0.20,
                "connectivity_score": 0.20,
                "link_density": 0.15,
                "authority_score": 0.20,
            },
        }

    def _get_default_normalization_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get default percentile-based normalization ranges.

        These represent (p10, p90) percentiles - the 10th and 90th percentile values
        from typical Wikipedia article data. This is more adaptive than hard-coded
        thresholds and can be recalibrated with real training data.

        Returns:
            Dictionary mapping feature names to (min_val, max_val) tuples.
        """
        return {
            # Structure features
            "section_count": (2, 25),  # p10=2, p90=25
            "content_length": (200, 3500),  # p10=200, p90=3500
            "template_count": (1, 15),  # p10=1, p90=15
            "avg_section_depth": (1.0, 2.5),  # p10=1.0, p90=2.5
            "sections_per_1k_chars": (0.5, 8.0),  # p10=0.5, p90=8.0
            # Sourcing features
            "citation_count": (1, 50),  # p10=1, p90=50
            "citations_per_1k_tokens": (0.5, 25),  # p10=0.5, p90=25
            "external_link_count": (1, 40),  # p10=1, p90=40
            "citation_density": (0.0001, 0.02),  # p10=0.0001, p90=0.02
            "academic_source_ratio": (0, 0.3),  # p10=0, p90=0.3
            # Editorial features
            "total_editors": (1, 25),  # p10=1, p90=25
            "total_revisions": (2, 100),  # p10=2, p90=100
            "editor_diversity": (0.1, 0.6),  # p10=0.1, p90=0.6
            "recent_activity_score": (0, 5),  # p10=0, p90=5
            "major_editor_ratio": (0, 0.5),  # p10=0, p90=0.5
            # Network features
            "inbound_links": (1, 60),  # p10=1, p90=60
            "outbound_links": (5, 150),  # p10=5, p90=150
            "connectivity_score": (0.2, 0.8),  # p10=0.2, p90=0.8
            "link_density": (0.0001, 0.01),  # p10=0.0001, p90=0.01
            "authority_score": (0, 1.0),  # p10=0, p90=1.0
        }

    def calibrate_normalization_ranges(
        self, features_list: List[Dict[str, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """Calibrate normalization ranges from actual feature data.

        This method computes percentile-based ranges from a collection of articles,
        making the normalization adaptive to real data distribution.

        Args:
            features_list: List of feature dictionaries extracted from articles.

        Returns:
            Dictionary mapping feature names to (p10, p90) value pairs.
        """
        if not features_list:
            return self._get_default_normalization_ranges()

        calibrated_ranges: Dict[str, Tuple[float, float]] = {}

        # Convert to DataFrame for easier percentile calculation
        features_df = None
        try:
            import pandas as pd

            features_df = pd.DataFrame(features_list)
        except ImportError:
            # Fallback to numpy if pandas not available
            features_dict: Dict[str, List[float]] = {}
            for feature_dict in features_list:
                for key, value in feature_dict.items():
                    if key not in features_dict:
                        features_dict[key] = []
                    features_dict[key].append(
                        float(value) if isinstance(value, (int, float)) else 0.0
                    )

        # Calculate p10 and p90 for each feature
        for feature_name in self.normalization_ranges.keys():
            try:
                if features_df is not None:
                    if feature_name in features_df.columns:
                        values = features_df[feature_name].dropna()
                        if len(values) > 0:
                            p10 = float(np.percentile(values, 10))
                            p90 = float(np.percentile(values, 90))
                            calibrated_ranges[feature_name] = (p10, p90)
                else:
                    # Using fallback dictionary
                    if feature_name in features_dict:
                        values = [
                            v for v in features_dict[feature_name] if v is not None
                        ]
                        if len(values) > 0:
                            p10 = float(np.percentile(values, 10))
                            p90 = float(np.percentile(values, 90))
                            calibrated_ranges[feature_name] = (p10, p90)
            except (ValueError, TypeError):
                # Use default if calculation fails
                pass

        # Fill in any missing features with defaults
        for (
            feature_name,
            default_range,
        ) in self._get_default_normalization_ranges().items():
            if feature_name not in calibrated_ranges:
                calibrated_ranges[feature_name] = default_range

        self.normalization_ranges = calibrated_ranges
        return calibrated_ranges

    def extract_features(self, article_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract all features from article data.

        Args:
            article_data: Raw Wikipedia article JSON data.

        Returns:
            Dictionary containing all extracted features.
        """
        features = {}

        # Extract features from each pillar
        features.update(structure_features(article_data))
        features.update(sourcing_features(article_data))
        features.update(editorial_features(article_data))
        features.update(network_features(article_data))

        return features

    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to 0-1 scale using percentile-based ranges.

        Uses adaptive normalization based on percentile ranges (p10 to p90) rather than
        hard-coded thresholds. This provides more balanced scoring across the full
        spectrum of article quality.

        Args:
            features: Raw feature values.

        Returns:
            Dictionary of normalized features.
        """
        normalized = {}

        for feature_name, value in features.items():
            if feature_name in self.normalization_ranges:
                p10_val, p90_val = self.normalization_ranges[feature_name]

                if p90_val > p10_val:
                    # Linear normalization to 0-1 scale
                    # Values at p10 map to ~0, values at p90 map to ~1
                    norm_value = (value - p10_val) / (p90_val - p10_val)

                    # Smooth sigmoid-like curve instead of hard boosting
                    # Values below p10 stay near 0, values above p90 approach 1
                    # No artificial amplification, natural distribution
                    normalized[feature_name] = max(0.0, min(1.0, norm_value))
                else:
                    # Degenerate range (p10 == p90).
                    # If it's a binary/ratio feature, just clamp to 0-1.
                    # Otherwise, if value equals the constant value, assume 1.0 (or 0.5? or 0.0?)
                    # For metrics where the "constant" is the ideal (e.g. 1.0 for has_infobox),
                    # matching it should be good. But generally, no variance = no information.
                    # However, for manual binary features, we trust the raw value more.
                    if feature_name.startswith("has_") or feature_name.endswith(
                        "_ratio"
                    ):
                        normalized[feature_name] = max(0.0, min(1.0, value))
                    else:
                        normalized[feature_name] = 0.0
            else:
                # For binary features or already normalized features
                if feature_name.startswith("has_") or feature_name.endswith("_ratio"):
                    normalized[feature_name] = max(0.0, min(1.0, value))
                else:
                    # Log scale normalization for large values
                    normalized[feature_name] = max(
                        0.0, min(1.0, math.log(max(value, 1)) / 10)
                    )

        return normalized

    def calculate_pillar_scores(
        self, normalized_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate pillar scores from normalized features.

        Args:
            normalized_features: Normalized feature values.

        Returns:
            Dictionary containing pillar scores.
        """
        pillar_scores = {}

        # Structure pillar
        structure_features_list = [
            "section_count",
            "content_length",
            "has_infobox",
            "template_count",
            "avg_section_depth",
            "sections_per_1k_chars",
            "has_references",
            "has_external_links",
        ]
        structure_score = self._calculate_weighted_score(
            normalized_features, structure_features_list, "structure"
        )
        pillar_scores["structure"] = structure_score

        # Sourcing pillar
        sourcing_features_list = [
            "citation_count",
            "citations_per_1k_tokens",
            "external_link_count",
            "citation_density",
            "has_reliable_sources",
            "academic_source_ratio",
        ]
        sourcing_score = self._calculate_weighted_score(
            normalized_features, sourcing_features_list, "sourcing"
        )
        pillar_scores["sourcing"] = sourcing_score

        # Editorial pillar
        editorial_features_list = [
            "total_editors",
            "total_revisions",
            "editor_diversity",
            "recent_activity_score",
            "bot_edit_ratio",
            "major_editor_ratio",
        ]
        editorial_score = self._calculate_weighted_score(
            normalized_features, editorial_features_list, "editorial"
        )
        pillar_scores["editorial"] = editorial_score

        # Network pillar
        network_features_list = [
            "inbound_links",
            "outbound_links",
            "connectivity_score",
            "link_density",
            "authority_score",
        ]
        network_score = self._calculate_weighted_score(
            normalized_features, network_features_list, "network"
        )
        pillar_scores["network"] = network_score

        return pillar_scores

    def _calculate_weighted_score(
        self, features: Dict[str, float], feature_list: List[str], pillar: str
    ) -> float:
        """Calculate weighted score for a pillar.

        Args:
            features: Normalized feature values.
            feature_list: List of features to include in calculation.
            pillar: Pillar name for weight lookup.

        Returns:
            Weighted score for the pillar.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for feature_name in feature_list:
            if feature_name in features:
                weight = self.feature_weights.get(feature_name, 0.1)
                weighted_sum += features[feature_name] * weight
                total_weight += abs(weight)  # Use absolute weight for normalization

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    def calculate_maturity_score(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall maturity score for an article.

        Args:
            article_data: Raw Wikipedia article JSON data.

        Returns:
            Dictionary containing maturity score and breakdown.
        """
        # Extract and normalize features
        raw_features = self.extract_features(article_data)
        normalized_features = self.normalize_features(raw_features)

        # Calculate pillar scores
        pillar_scores = self.calculate_pillar_scores(normalized_features)

        # Calculate overall maturity score
        # Calculate weighted average of pillar scores
        # No artificial bonuses - straightforward and transparent
        maturity_score = 0.0
        total_weight = 0.0

        for pillar, score in pillar_scores.items():
            weight = self.pillar_weights.get(pillar, 0.25)
            maturity_score += score * weight
            total_weight += weight

        if total_weight > 0:
            maturity_score = maturity_score / total_weight

        # Apply stub penalty to prevent gaming density metrics
        # Rationale: Articles with minimal content can have artificially high
        # density metrics (citations per token, link density) despite poor quality
        content_length = raw_features.get("content_length", 0)
        section_count = raw_features.get("section_count", 0)

        stub_penalty = self._calculate_continuous_stub_penalty(
            content_length, section_count
        )

        maturity_score = maturity_score * stub_penalty

        # Scale to 0-100 range
        maturity_score = maturity_score * 100
        scaled_pillar_scores = {k: round(v * 100, 2) for k, v in pillar_scores.items()}

        return {
            "maturity_score": round(maturity_score, 2),
            "pillar_scores": scaled_pillar_scores,
            "raw_features": raw_features,
            "normalized_features": normalized_features,
            "weights_used": {
                "pillars": self.pillar_weights,
                "features": self.feature_weights,
            },
        }

    def _calculate_continuous_stub_penalty(
        self, content_length: float, section_count: float
    ) -> float:
        """Calculate a continuous stub penalty.

        Uses a logistic function to provide a smooth transition from penalized to
        unpenalized states, avoiding cliffs in the scoring function.

        Args:
            content_length: Length of content in characters.
            section_count: Number of sections.

        Returns:
            Penalty factor between 0.0 (severe penalty) and 1.0 (no penalty).
        """
        # Content length sigmoid
        # Center at 1500 chars, slope 0.003
        # < 500 chars -> ~0.05
        # 1500 chars -> 0.5
        # > 3000 chars -> ~0.99
        len_score = 1 / (1 + math.exp(-0.003 * (content_length - 1500)))

        # Section count sigmoid
        # Center at 6 sections, slope 0.8
        # < 2 sections -> ~0.04
        # 6 sections -> 0.5
        # > 10 sections -> ~0.96
        sec_score = 1 / (1 + math.exp(-0.8 * (section_count - 6)))

        # Combined score, biased towards the lower of the two (so you need both)
        # Using harmonic mean interaction or just geometric mean
        # Here we use a weighted geometric mean, slightly favoring content length
        combined_score = (len_score**0.6) * (sec_score**0.4)

        # Map to penalty range [0.5, 1.0]
        # We don't want to go below 0.5 even for empty articles (mostly likely)
        # to avoid completely zeroing out other good metrics if they exist
        penalty = 0.5 + 0.5 * combined_score

        return float(min(1.0, max(0.0, penalty)))

    def calibrate_weights(
        self,
        training_data: List[Tuple[Dict[str, Any], float]],
        target_correlation: float = 0.6,
    ) -> Dict[str, Any]:
        """Calibrate weights using training data to achieve target correlation.

        First calibrates normalization ranges from feature data, then optimizes
        pillar weights using scipy.optimize to maximize correlation with target scores.

        Args:
            training_data: List of (article_data, target_score) tuples.
            target_correlation: Target correlation with ground truth scores.

        Returns:
            Dictionary containing calibrated weights and correlation metrics.
        """
        if not training_data:
            return {
                "best_correlation": 0.0,
                "calibrated_weights": self.weights,
                "target_correlation": target_correlation,
                "message": "No training data provided",
            }

        # First pass: calibrate normalization ranges from all features
        all_features = [article_data for article_data, _ in training_data]
        raw_features_list = [
            self.extract_features(article_data) for article_data in all_features
        ]
        self.calibrate_normalization_ranges(raw_features_list)

        # Prepare data for optimization
        # Pre-calculate normalized features to avoid re-computing in loop
        normalized_data: List[Dict[str, Any]] = []
        target_scores = []

        for i, article_data in enumerate(all_features):
            raw = raw_features_list[i]
            norm = self.normalize_features(raw)
            # Pre-calculate pillar raw scores (without weights)
            # This is complex because pillar scores depend on feature weights,
            # which we are NOT optimizing here (yet). We are optimizing pillar weights.
            pillar_scores = self.calculate_pillar_scores(norm)

            # We also need stub penalty
            stub_penalty = self._calculate_continuous_stub_penalty(
                raw.get("content_length", 0), raw.get("section_count", 0)
            )

            normalized_data.append(
                {"pillar_scores": pillar_scores, "stub_penalty": stub_penalty}
            )
            target_scores.append(training_data[i][1])

        # Objective function to MINIMIZE (negative correlation)
        def objective(weights_array: np.ndarray) -> float:
            # weights_array is [w_structure, w_sourcing, w_editorial, w_network]
            # Normalize to sum to 1.0 logic is handled by constraints, but good to be safe

            w_struct, w_sourc, w_edit, w_net = weights_array

            predicted = []
            for item in normalized_data:
                p_scores = item["pillar_scores"]
                # Weighted average
                raw_score = (
                    p_scores.get("structure", 0) * w_struct
                    + p_scores.get("sourcing", 0) * w_sourc
                    + p_scores.get("editorial", 0) * w_edit
                    + p_scores.get("network", 0) * w_net
                )

                # Normalize by sum of weights (should be ~1.0)
                weight_sum = w_struct + w_sourc + w_edit + w_net
                if weight_sum > 0:
                    raw_score /= weight_sum

                # Apply penalty
                final_score = raw_score * item["stub_penalty"] * 100
                predicted.append(final_score)

            # Calculate correlation
            if len(predicted) < 2:
                return 0.0

            try:
                # We want to MAXIMIZE correlation, so MINIMIZE negative correlation
                corr_matrix = np.corrcoef(predicted, target_scores)
                # handle NaN
                if np.isnan(corr_matrix).any():
                    return 0.0
                return float(-corr_matrix[0, 1])
            except Exception:
                return 0.0

        # Initial weights
        initial_weights = [
            self.pillar_weights.get("structure", 0.25),
            self.pillar_weights.get("sourcing", 0.25),
            self.pillar_weights.get("editorial", 0.25),
            self.pillar_weights.get("network", 0.25),
        ]

        # Constraints: sum to 1.0
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

        # Bounds: each weight between 0.05 (min relevance) and 0.6
        bounds = [(0.05, 0.6) for _ in range(4)]

        # Optimization
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        best_correlation = -result.fun if result.success else 0.0

        # Update weights if successful
        if result.success and best_correlation > 0:
            optimized_weights = result.x
            best_weights = self.weights.copy()
            best_weights["pillars"] = {
                "structure": float(optimized_weights[0]),
                "sourcing": float(optimized_weights[1]),
                "editorial": float(optimized_weights[2]),
                "network": float(optimized_weights[3]),
            }
            self.weights = best_weights
            self.pillar_weights = best_weights["pillars"]
            self.save_weights()

        return {
            "best_correlation": best_correlation,
            "calibrated_weights": self.weights,
            "target_correlation": target_correlation,
            "normalization_ranges": self.normalization_ranges,
            "optimization_success": result.success,
            "message": result.message,
        }

    def save_weights(self, output_file: Optional[str] = None) -> None:
        """Save current weights and normalization ranges to YAML file.

        Args:
            output_file: Output file path. If None, uses current weights_file.
        """
        output_path = output_file or self.weights_file

        # Include normalization ranges in saved weights
        weights_to_save = self.weights.copy()
        weights_to_save["normalization_ranges"] = {
            k: list(v) for k, v in self.normalization_ranges.items()
        }

        with open(output_path, "w") as f:
            yaml.dump(weights_to_save, f, default_flow_style=False, indent=2)

    def get_feature_importance(self, article_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature importance for an article.

        Args:
            article_data: Raw Wikipedia article JSON data.

        Returns:
            Dictionary containing feature importance scores.
        """
        # Extract features
        raw_features = self.extract_features(article_data)
        normalized_features = self.normalize_features(raw_features)

        # Calculate importance based on weights and feature values
        importance = {}

        for feature_name, value in normalized_features.items():
            weight = self.feature_weights.get(feature_name, 0.1)
            importance[feature_name] = abs(weight * value)

        # Sort by importance
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance
