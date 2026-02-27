"""Heuristic baseline model for Wikipedia article maturity scoring.

This module implements a weighted average approach to calculate maturity scores
based on normalized features from structural, sourcing, editorial, and network
pillars. The weights are configurable via YAML and can be calibrated using
GA/FA vs Stub/Start article examples.

Features are normalized using percentile-based ranges derived from training data,
enabling adaptive normalization that reflects actual feature distributions.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from wikipedia.features.graph_processor import GraphProcessor

import yaml  # type: ignore


class HeuristicBaselineModel:
    """Heuristic baseline model for Wikipedia article maturity scoring.

    This model calculates maturity scores using a weighted average of normalized
    features across four pillars: structure, sourcing, editorial, and network.
    The weights are configurable and transparent.
    """

    def __init__(
        self,
        weights_file: Optional[str] = None,
        normalization_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """Initialize the heuristic baseline model."""
        self.weights_file = weights_file or "wikipedia/models/weights.yaml"
        self.normalization_ranges: Dict[str, Tuple[float, float]] = (
            normalization_ranges or self._get_default_normalization_ranges()
        )
        self.weights = self._load_weights()
        self.pillar_weights = self.weights.get("pillars", {})
        self.feature_weights = self.weights.get("features", {})

    def _load_weights(self) -> Dict[str, Any]:
        """Load weights from YAML configuration file."""
        weights_path = Path(self.weights_file)

        if not weights_path.exists():
            return self._get_default_weights()

        try:
            with open(weights_path, "r") as f:
                weights = yaml.safe_load(f)
            return weights or self._get_default_weights()
        except Exception:
            return self._get_default_weights()

    def _get_default_weights(self) -> Dict[str, Any]:
        """Get default weights for the heuristic model."""
        return {
            "pillars": {
                "structure": 0.25,
                "sourcing": 0.30,
                "editorial": 0.25,
                "network": 0.20,
            },
            "features": {
                "section_count": 0.15,
                "content_length": 0.20,
                "has_infobox": 0.10,
                "template_count": 0.10,
                "avg_section_depth": 0.10,
                "sections_per_1k_chars": 0.10,
                "has_references": 0.10,
                "has_external_links": 0.15,
                "citation_count": 0.25,
                "citations_per_1k_tokens": 0.20,
                "external_link_count": 0.15,
                "citation_density": 0.15,
                "has_reliable_sources": 0.15,
                "academic_source_ratio": 0.10,
                "total_editors": 0.20,
                "total_revisions": 0.20,
                "editor_diversity": 0.15,
                "recent_activity_score": 0.15,
                "bot_edit_ratio": -0.10,
                "major_editor_ratio": 0.20,
                "inbound_links": 0.25,
                "outbound_links": 0.20,
                "connectivity_score": 0.20,
                "link_density": 0.15,
                "authority_score": 0.20,
            },
        }

    def _get_default_normalization_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get default normalization ranges."""
        return {
            "section_count": (2, 25),
            "content_length": (200, 3500),
            "template_count": (1, 15),
            "avg_section_depth": (1.0, 2.5),
            "sections_per_1k_chars": (0.5, 8.0),
            "citation_count": (1, 50),
            "citations_per_1k_tokens": (0.5, 25),
            "external_link_count": (1, 40),
            "citation_density": (0.0001, 0.02),
            "academic_source_ratio": (0, 0.3),
            "total_editors": (1, 25),
            "total_revisions": (2, 100),
            "editor_diversity": (0.1, 0.6),
            "recent_activity_score": (0, 5),
            "major_editor_ratio": (0, 0.5),
            "inbound_links": (1, 60),
            "outbound_links": (5, 150),
            "connectivity_score": (0.2, 0.8),
            "link_density": (0.0001, 0.01),
            "authority_score": (0, 1.0),
            "hub_score": (0, 1.0),
            "link_balance": (0, 1.0),
            "network_centrality": (0, 1.0),
        }

    def extract_features(
        self,
        article_data: Dict[str, Any],
        graph_processor: Optional["GraphProcessor"] = None,
    ) -> Dict[str, float]:
        """Extract all features from article data, including global metrics if available.

        Args:
            article_data: Raw Wikipedia article JSON data.
            graph_processor: Optional pre-computed graph metrics.

        Returns:
            Dictionary containing all extracted features.
        """
        from wikipedia.features.extractors import all_features

        return all_features(article_data, graph_processor=graph_processor)

    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Simple linear normalization to 0-1 scale."""
        normalized = {}
        for name, val in features.items():
            if name in self.normalization_ranges:
                low, high = self.normalization_ranges[name]
                if high > low:
                    normalized[name] = max(0.0, min(1.0, (val - low) / (high - low)))
                else:
                    normalized[name] = 1.0 if val >= low else 0.0
            else:
                # Fallback for ratios or indicators
                normalized[name] = max(0.0, min(1.0, val))
        return normalized

    def calculate_pillar_scores(
        self, norm_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate score for each pillar."""
        pillars = {
            "structure": [
                "section_count",
                "content_length",
                "has_infobox",
                "template_count",
                "avg_section_depth",
                "sections_per_1k_chars",
                "has_references",
                "has_external_links",
            ],
            "sourcing": [
                "citation_count",
                "citations_per_1k_tokens",
                "external_link_count",
                "citation_density",
                "has_reliable_sources",
                "academic_source_ratio",
            ],
            "editorial": [
                "total_editors",
                "total_revisions",
                "editor_diversity",
                "recent_activity_score",
                "bot_edit_ratio",
                "major_editor_ratio",
            ],
            "network": [
                "inbound_links",
                "outbound_links",
                "connectivity_score",
                "link_density",
                "authority_score",
            ],
        }

        scores = {}
        for pillar, feats in pillars.items():
            total_weight = 0.0
            weighted_sum = 0.0
            for f in feats:
                if f in norm_features:
                    w = self.feature_weights.get(f, 0.1)
                    weighted_sum += norm_features[f] * w
                    total_weight += abs(w)
            scores[pillar] = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        return scores

    def calculate_maturity_score(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for scoring an article."""
        from wikipedia.features.extractors import all_features

        raw = all_features(article_data)
        norm = self.normalize_features(raw)
        pillars = self.calculate_pillar_scores(norm)

        # Weighted average of pillars
        score = 0.0
        total_w = 0.0
        for p, s in pillars.items():
            w = self.pillar_weights.get(p, 0.25)
            score += s * w
            total_w += w

        final_score = (score / total_w) if total_w > 0 else 0.0

        # Simple stub penalty
        content_len = raw.get("content_length", 0)
        if content_len < 1000:
            final_score *= 0.5 + 0.5 * (content_len / 1000)

        return {
            "maturity_score": round(final_score * 100, 2),
            "pillar_scores": {k: round(v * 100, 2) for k, v in pillars.items()},
            "raw_features": raw,
        }

    def get_feature_importance(self, article_data: Dict[str, Any]) -> Dict[str, float]:
        """Simple feature importance based on weights."""
        from wikipedia.features.extractors import all_features

        raw = all_features(article_data)
        norm = self.normalize_features(raw)

        importance = {f: abs(norm[f] * self.feature_weights.get(f, 0.1)) for f in norm}
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
