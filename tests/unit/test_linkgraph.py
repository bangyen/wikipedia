#!/usr/bin/env python3
"""Unit tests for link graph feature extraction.

This module tests the link graph feature extraction functionality,
including PageRank, betweenness centrality, and orphan detection metrics.
"""

import math
import pytest

from wikipedia.features.linkgraph import linkgraph_features


class TestLinkGraphFeatures:
    """Test cases for link graph feature extraction."""

    def test_empty_article_data(self) -> None:
        """Test link graph features with empty article data."""
        article_data = {"title": "Test Article", "data": {}}

        features = linkgraph_features(article_data)

        # Should return zero values for most features, but centrality measures are 1.0 for single node
        assert features["pagerank_score"] == 1.0
        assert features["betweenness_centrality"] == 0.0
        assert (
            features["degree_centrality"] == 1.0
        )  # Single node has degree centrality 1.0
        assert features["closeness_centrality"] == 0.0
        assert (
            features["eigenvector_centrality"] == 1.0
        )  # Single node has eigenvector centrality 1.0
        assert features["clustering_coefficient"] == 0.0
        assert features["orphan_score"] == 1.0
        assert features["hub_score"] == 0.0
        assert features["authority_score"] == 0.0
        assert features["connectivity_ratio"] == 0.0
        assert features["structural_holes"] == 0.0
        assert features["core_periphery_score"] == 0.0
        assert features["isolation_score"] == 1.0
        assert features["dead_end_score"] == 0.0
        assert features["graph_density"] == 0.0
        assert features["assortativity"] == 0.0
        assert features["small_world_coefficient"] == 0.0
        assert (
            features["log_pagerank_score"] == -23.025850929940457
        )  # log(1e-10) for very small PageRank
        assert (
            features["log_degree_centrality"] == -23.025850929940457
        )  # log(1e-10) for very small degree centrality
        assert features["link_balance"] == 0.0

    def test_article_with_backlinks(self) -> None:
        """Test link graph features with backlinks."""
        article_data = {
            "title": "Test Article",
            "data": {
                "query": {
                    "backlinks": [
                        {"title": "Source Article 1"},
                        {"title": "Source Article 2"},
                        {"title": "Source Article 3"},
                    ],
                    "pages": {
                        "12345": {
                            "links": [
                                {"title": "Target Article 1"},
                                {"title": "Target Article 2"},
                            ]
                        }
                    },
                }
            },
        }

        features = linkgraph_features(article_data)

        # Should have non-zero values for connected articles
        assert features["pagerank_score"] > 0.0
        assert features["degree_centrality"] > 0.0
        assert features["orphan_score"] < 1.0
        assert features["hub_score"] >= 0.0
        assert features["authority_score"] >= 0.0
        assert features["connectivity_ratio"] > 0.0
        assert features["link_balance"] >= 0.0

    def test_orphan_article(self) -> None:
        """Test link graph features for orphaned article."""
        article_data = {
            "title": "Orphan Article",
            "data": {"query": {"backlinks": [], "pages": {"12345": {"links": []}}}},
        }

        features = linkgraph_features(article_data)

        # Should be completely orphaned
        assert features["orphan_score"] == 1.0
        assert features["isolation_score"] == 1.0
        assert features["pagerank_score"] == 1.0  # Single node gets PageRank 1.0
        assert (
            features["degree_centrality"] == 1.0
        )  # Single node has degree centrality 1.0
        assert features["connectivity_ratio"] == 0.0

    def test_hub_article(self) -> None:
        """Test link graph features for hub-like article."""
        article_data = {
            "title": "Hub Article",
            "data": {
                "query": {
                    "backlinks": [
                        {"title": "Source Article 1"},
                    ],
                    "pages": {
                        "12345": {
                            "links": [
                                {"title": "Target Article 1"},
                                {"title": "Target Article 2"},
                                {"title": "Target Article 3"},
                                {"title": "Target Article 4"},
                                {"title": "Target Article 5"},
                            ]
                        }
                    },
                }
            },
        }

        features = linkgraph_features(article_data)

        # Should have high hub score (more outgoing than incoming)
        assert features["hub_score"] > features["authority_score"]
        assert features["hub_score"] > 0.5
        assert features["authority_score"] < 0.5

    def test_authority_article(self) -> None:
        """Test link graph features for authority-like article."""
        article_data = {
            "title": "Authority Article",
            "data": {
                "query": {
                    "backlinks": [
                        {"title": "Source Article 1"},
                        {"title": "Source Article 2"},
                        {"title": "Source Article 3"},
                        {"title": "Source Article 4"},
                        {"title": "Source Article 5"},
                    ],
                    "pages": {
                        "12345": {
                            "links": [
                                {"title": "Target Article 1"},
                            ]
                        }
                    },
                }
            },
        }

        features = linkgraph_features(article_data)

        # Should have high authority score (more incoming than outgoing)
        # Note: hub_score = out_degree / (in_degree + out_degree), authority_score = in_degree / (in_degree + out_degree)
        # With 5 incoming and 1 outgoing: hub_score = 1/6, authority_score = 5/6
        assert features["authority_score"] > features["hub_score"]
        assert features["authority_score"] > 0.5
        assert features["hub_score"] < 0.5

    def test_feature_types(self) -> None:
        """Test that all features are numeric."""
        article_data = {
            "title": "Test Article",
            "data": {
                "query": {
                    "backlinks": [{"title": "Source Article"}],
                    "pages": {"12345": {"links": [{"title": "Target Article"}]}},
                }
            },
        }

        features = linkgraph_features(article_data)

        # All features should be numeric
        for feature_name, feature_value in features.items():
            assert isinstance(
                feature_value, (int, float)
            ), f"{feature_name} should be numeric"
            assert not math.isnan(feature_value), f"{feature_name} should not be NaN"
            assert not math.isinf(
                feature_value
            ), f"{feature_name} should not be infinite"

    def test_feature_ranges(self) -> None:
        """Test that features are within expected ranges."""
        article_data = {
            "title": "Test Article",
            "data": {
                "query": {
                    "backlinks": [{"title": "Source Article"}],
                    "pages": {"12345": {"links": [{"title": "Target Article"}]}},
                }
            },
        }

        features = linkgraph_features(article_data)

        # Test specific feature ranges
        assert 0.0 <= features["pagerank_score"] <= 1.0
        assert 0.0 <= features["degree_centrality"] <= 1.0
        assert 0.0 <= features["betweenness_centrality"] <= 1.0
        assert 0.0 <= features["closeness_centrality"] <= 1.0
        assert 0.0 <= features["eigenvector_centrality"] <= 1.0
        assert 0.0 <= features["clustering_coefficient"] <= 1.0
        assert 0.0 <= features["orphan_score"] <= 1.0
        assert 0.0 <= features["hub_score"] <= 1.0
        assert 0.0 <= features["authority_score"] <= 1.0
        assert 0.0 <= features["connectivity_ratio"] <= 1.0
        assert 0.0 <= features["structural_holes"] <= 1.0
        assert 0.0 <= features["core_periphery_score"] <= 1.0
        assert 0.0 <= features["isolation_score"] <= 1.0
        assert 0.0 <= features["dead_end_score"] <= 1.0
        assert 0.0 <= features["graph_density"] <= 1.0
        assert -1.0 <= features["assortativity"] <= 1.0
        assert features["small_world_coefficient"] >= 0.0
        assert (
            features["log_pagerank_score"] >= -25.0
        )  # Allow negative log values for very small PageRank
        assert (
            features["log_degree_centrality"] >= -25.0
        )  # Allow negative log values for very small degree centrality
        assert 0.0 <= features["link_balance"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
