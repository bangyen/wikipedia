"""Heuristic baseline model for Wikipedia article maturity scoring.

This module implements a weighted average approach to calculate maturity scores
based on normalized features from structural, sourcing, editorial, and network
pillars. The weights are configurable via YAML and can be calibrated using
GA/FA vs Stub/Start article examples.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml  # type: ignore

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
    """

    def __init__(self, weights_file: Optional[str] = None) -> None:
        """Initialize the heuristic baseline model.

        Args:
            weights_file: Path to YAML file containing feature weights.
                         If None, uses default weights.
        """
        self.weights_file = weights_file or "models/weights.yaml"
        self.weights = self._load_weights()
        self.pillar_weights = self.weights.get("pillars", {})
        self.feature_weights = self.weights.get("features", {})

    def _load_weights(self) -> Dict[str, Any]:
        """Load weights from YAML configuration file.

        Returns:
            Dictionary containing pillar and feature weights.

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
        """Normalize features to 0-1 scale for scoring.

        Args:
            features: Raw feature values.

        Returns:
            Dictionary of normalized features.
        """
        normalized = {}

        # Define normalization ranges for key features
        # Optimized for Featured/Good article detection
        # Conservative ranges: reaching 80%+ indicates excellent quality
        normalization_ranges = {
            # Structure features
            "section_count": (
                0,
                55,
            ),  # 55+ sections = excellent (featured articles have 50-100+)
            "content_length": (0, 4500),  # Extract only, 4.5k+ = comprehensive
            "template_count": (0, 20),  # 20+ templates = well-formatted
            "avg_section_depth": (1, 4),  # Depth 3-4 = good organization
            "sections_per_1k_chars": (0, 15),
            # Sourcing features (fetch limit 100)
            "citation_count": (0, 70),  # 70+ = excellent sourcing
            "citations_per_1k_tokens": (0, 70),  # 70+ per 1k = very well sourced
            "external_link_count": (0, 70),  # 70+ external links = comprehensive
            "citation_density": (0, 0.05),
            "academic_source_ratio": (0, 0.5),  # 50%+ academic = exceptional
            # Editorial features (fetch limit 100)
            "total_editors": (1, 35),  # 35+ editors = highly collaborative
            "total_revisions": (1, 70),  # 70+ revisions in sample = very active
            "editor_diversity": (0, 1),
            "recent_activity_score": (0, 10),
            "major_editor_ratio": (0, 1),
            # Network features (fetch limit 100/200)
            "inbound_links": (0, 70),  # 70+ backlinks = highly connected
            "outbound_links": (0, 130),  # 130+ outbound = comprehensive
            "connectivity_score": (0, 1),
            "link_density": (0, 0.02),
            "authority_score": (0, 1),
        }

        for feature_name, value in features.items():
            if feature_name in normalization_ranges:
                min_val, max_val = normalization_ranges[feature_name]
                if max_val > min_val:
                    norm_value = (value - min_val) / (max_val - min_val)
                    # Give bonus credit for exceeding expected ranges (hitting API limits)
                    # Articles that max out are likely even better than we can measure
                    if norm_value >= 1.0:  # Maxed out
                        norm_value = 1.0  # Full credit
                    elif norm_value > 0.8:  # Near max (80%+)
                        # Boost high performers with a curve
                        norm_value = (
                            0.8 + (norm_value - 0.8) * 1.25
                        )  # 25% boost in top range
                    normalized[feature_name] = max(0, min(1, norm_value))
                else:
                    normalized[feature_name] = 0.0
            else:
                # For binary features or already normalized features
                if feature_name.startswith("has_") or feature_name.endswith("_ratio"):
                    normalized[feature_name] = max(0, min(1, value))
                else:
                    # Log scale normalization for large values
                    normalized[feature_name] = max(
                        0, min(1, math.log(max(value, 1)) / 10)
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
        maturity_score = 0.0
        total_weight = 0.0

        for pillar, score in pillar_scores.items():
            weight = self.pillar_weights.get(pillar, 0.25)
            maturity_score += score * weight
            total_weight += weight

        if total_weight > 0:
            maturity_score = maturity_score / total_weight

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

    def calibrate_weights(
        self,
        training_data: List[Tuple[Dict[str, Any], float]],
        target_correlation: float = 0.6,
    ) -> Dict[str, Any]:
        """Calibrate weights using training data to achieve target correlation.

        Args:
            training_data: List of (article_data, target_score) tuples.
            target_correlation: Target correlation with ground truth scores.

        Returns:
            Dictionary containing calibrated weights.
        """
        # Simple grid search for weight optimization
        best_weights = self.weights.copy()
        best_correlation = 0.0

        # Test different pillar weight combinations
        pillar_combinations = [
            {"structure": 0.3, "sourcing": 0.4, "editorial": 0.2, "network": 0.1},
            {"structure": 0.25, "sourcing": 0.35, "editorial": 0.25, "network": 0.15},
            {"structure": 0.2, "sourcing": 0.3, "editorial": 0.3, "network": 0.2},
            {"structure": 0.3, "sourcing": 0.3, "editorial": 0.2, "network": 0.2},
        ]

        for pillar_weights in pillar_combinations:
            # Temporarily update weights
            old_weights = self.pillar_weights.copy()
            self.pillar_weights = pillar_weights

            # Calculate correlation
            predicted_scores = []
            target_scores = []

            for article_data, target_score in training_data:
                result = self.calculate_maturity_score(article_data)
                predicted_scores.append(result["maturity_score"])
                target_scores.append(target_score)

            if len(predicted_scores) > 1:
                correlation_matrix = np.corrcoef(predicted_scores, target_scores)
                correlation = correlation_matrix[0, 1]

                if abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_weights["pillars"] = pillar_weights.copy()

            # Restore original weights
            self.pillar_weights = old_weights

        # Update weights if we found a better combination
        if abs(best_correlation) > 0.1:  # Only update if correlation is meaningful
            self.weights = best_weights
            self.pillar_weights = best_weights["pillars"]

            # Save calibrated weights
            self.save_weights()

        return {
            "best_correlation": best_correlation,
            "calibrated_weights": best_weights,
            "target_correlation": target_correlation,
        }

    def save_weights(self, output_file: Optional[str] = None) -> None:
        """Save current weights to YAML file.

        Args:
            output_file: Output file path. If None, uses current weights_file.
        """
        output_path = output_file or self.weights_file

        with open(output_path, "w") as f:
            yaml.dump(self.weights, f, default_flow_style=False, indent=2)

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
