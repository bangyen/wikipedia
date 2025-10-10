"""Unit tests for feature extraction functions."""

import math
from datetime import datetime, timezone, timedelta

from typing import Any, Dict

from features.extractors import (
    structure_features,
    sourcing_features,
    editorial_features,
    network_features,
)


class TestStructureFeatures:
    """Test cases for structure_features function."""

    def test_structure_features_basic(self) -> None:
        """Test basic structure feature extraction."""
        article_data: Dict[str, Any] = {
            "data": {
                "parse": {
                    "sections": [
                        {"index": 1, "line": "Introduction", "level": 1},
                        {"index": 2, "line": "Early Life", "level": 2},
                        {"index": 3, "line": "Career", "level": 2},
                        {"index": 4, "line": "Legacy", "level": 1},
                    ],
                    "text": {"*": "This is a test article with some content. " * 100},
                }
            }
        }

        features = structure_features(article_data)

        # Basic counts
        assert features["section_count"] == 4.0
        assert features["content_length"] > 0
        assert features["avg_section_depth"] == 1.5
        assert features["max_section_depth"] == 2.0
        assert features["min_section_depth"] == 1.0

        # Normalized metrics
        assert features["sections_per_1k_chars"] > 0
        assert features["log_content_length"] > 0
        assert features["log_section_count"] > 0

        # Template flags
        assert features["has_infobox"] == 0.0
        assert features["has_categories"] == 0.0
        assert features["has_navbox"] == 0.0

    def test_structure_features_with_templates(self) -> None:
        """Test structure features with templates."""
        article_data: Dict[str, Any] = {
            "data": {
                "query": {
                    "pages": {
                        "123": {
                            "templates": [
                                {"title": "Template:Infobox scientist"},
                                {"title": "Template:Authority control"},
                                {"title": "Template:Navbox"},
                            ]
                        }
                    }
                },
                "parse": {
                    "sections": [
                        {"index": 1, "line": "Introduction", "level": 1},
                    ],
                    "text": {"*": "Test content"},
                },
            }
        }

        features = structure_features(article_data)

        assert features["template_count"] == 3.0
        assert features["has_infobox"] == 1.0
        assert features["has_navbox"] == 1.0

    def test_structure_features_empty(self) -> None:
        """Test structure features with empty data."""
        article_data: Dict[str, Any] = {"data": {}}

        features = structure_features(article_data)

        assert features["section_count"] == 0.0
        assert features["template_count"] == 0.0
        assert features["content_length"] == 0.0
        assert features["avg_section_depth"] == 0.0
        assert features["max_section_depth"] == 0.0
        assert features["sections_per_1k_chars"] == 0.0


class TestSourcingFeatures:
    """Test cases for sourcing_features function."""

    def test_sourcing_features_basic(self) -> None:
        """Test basic sourcing feature extraction."""
        article_data: Dict[str, Any] = {
            "data": {
                "query": {
                    "pages": {
                        "123": {
                            "extlinks": [
                                {"url": "https://example.com/source1"},
                                {"url": "https://harvard.edu/research"},
                                {"url": "https://bbc.com/news"},
                                {"url": "https://nasa.gov/data"},
                            ]
                        }
                    }
                },
                "parse": {
                    "text": {"*": "This is a test article with some content. " * 50}
                },
            }
        }

        features = sourcing_features(article_data)

        # Basic counts
        assert features["citation_count"] == 4.0
        assert features["external_link_count"] == 4.0
        assert features["token_count"] > 0

        # Normalized metrics
        assert features["citations_per_1k_tokens"] > 0
        assert features["external_links_per_1k_tokens"] > 0
        assert features["citation_density"] > 0

        # Source type ratios
        assert features["academic_source_ratio"] > 0  # harvard.edu
        assert features["news_source_ratio"] > 0  # bbc.com
        assert features["gov_source_ratio"] > 0  # nasa.gov
        assert features["org_source_ratio"] == 0.0  # example.com is .com, not .org

        # Log scaling
        assert features["log_citation_count"] > 0
        assert features["log_external_link_count"] > 0

    def test_sourcing_features_empty(self) -> None:
        """Test sourcing features with empty data."""
        article_data: Dict[str, Any] = {"data": {}}

        features = sourcing_features(article_data)

        assert features["citation_count"] == 0.0
        assert features["external_link_count"] == 0.0
        assert features["token_count"] == 0.0
        assert features["citations_per_1k_tokens"] == 0.0
        assert features["citation_density"] == 0.0
        assert features["academic_source_ratio"] == 0.0


class TestEditorialFeatures:
    """Test cases for editorial_features function."""

    def test_editorial_features_basic(self) -> None:
        """Test basic editorial feature extraction."""
        now = datetime.now(timezone.utc)

        article_data: Dict[str, Any] = {
            "data": {
                "query": {
                    "pages": {
                        "123": {
                            "revisions": [
                                {
                                    "revid": 1,
                                    "timestamp": (now - timedelta(days=5)).isoformat(),
                                    "user": "Editor1",
                                    "comment": "Test edit",
                                    "size": 1000,
                                },
                                {
                                    "revid": 2,
                                    "timestamp": (now - timedelta(days=15)).isoformat(),
                                    "user": "Editor2",
                                    "comment": "Another edit",
                                    "size": 1200,
                                },
                                {
                                    "revid": 3,
                                    "timestamp": (
                                        now - timedelta(days=100)
                                    ).isoformat(),
                                    "user": "Editor1",
                                    "comment": "Old edit",
                                    "size": 800,
                                },
                                {
                                    "revid": 4,
                                    "timestamp": (now - timedelta(days=1)).isoformat(),
                                    "user": "BotUser",
                                    "comment": "Bot edit",
                                    "size": 900,
                                },
                            ]
                        }
                    }
                }
            }
        }

        features = editorial_features(article_data)

        # Basic counts
        assert features["total_editors"] == 3.0  # Editor1, Editor2, BotUser
        assert features["total_revisions"] == 4.0

        # Time-based counts
        assert features["editors_90_days"] >= 2.0  # Editor1, Editor2
        assert features["editors_30_days"] >= 2.0  # Editor1, Editor2
        assert features["editors_7_days"] >= 1.0  # Editor1

        # Ratios
        assert features["bot_edit_ratio"] == 0.25  # 1 out of 4 edits
        assert features["anonymous_edit_ratio"] == 0.0

        # Other metrics
        assert features["revisions_per_editor"] > 0
        assert features["editor_diversity"] >= 0.0
        assert features["recent_activity_score"] > 0

        # Log scaling
        assert features["log_total_editors"] > 0
        assert features["log_total_revisions"] > 0

    def test_editorial_features_empty(self) -> None:
        """Test editorial features with empty data."""
        article_data: Dict[str, Any] = {"data": {}}

        features = editorial_features(article_data)

        assert features["total_editors"] == 0.0
        assert features["total_revisions"] == 0.0
        assert features["editors_90_days"] == 0.0
        assert features["bot_edit_ratio"] == 0.0
        assert features["revisions_per_editor"] == 0.0


class TestNetworkFeatures:
    """Test cases for network_features function."""

    def test_network_features_basic(self) -> None:
        """Test basic network feature extraction."""
        article_data: Dict[str, Any] = {
            "data": {
                "query": {
                    "backlinks": [
                        {"title": "Physics", "ns": 0},
                        {"title": "Theory of relativity", "ns": 0},
                        {"title": "Science", "ns": 0},
                    ],
                    "pages": {
                        "123": {
                            "links": [
                                {"title": "Mathematics", "ns": 0},
                                {"title": "Chemistry", "ns": 0},
                            ],
                            "extlinks": [
                                {"url": "https://example.com"},
                                {"url": "https://test.org"},
                            ],
                        }
                    },
                },
                "parse": {"text": {"*": "Test content with links"}},
            }
        }

        features = network_features(article_data)

        # Basic counts
        assert features["inbound_links"] == 3.0
        assert features["outbound_links"] == 2.0
        assert features["internal_links"] == 2.0
        assert features["external_links"] == 2.0
        assert features["total_links"] == 5.0

        # Connectivity metrics
        assert features["link_density"] > 0
        assert features["connectivity_score"] > 0
        assert features["hub_score"] > 0
        assert features["authority_score"] > 0
        assert features["link_balance"] > 0
        assert features["network_centrality"] > 0

        # Ratios
        assert features["internal_link_ratio"] > 0
        assert features["external_link_ratio"] > 0

        # Log scaling
        assert features["log_inbound_links"] > 0
        assert features["log_outbound_links"] > 0
        assert features["log_total_links"] > 0

    def test_network_features_empty(self) -> None:
        """Test network features with empty data."""
        article_data: Dict[str, Any] = {"data": {}}

        features = network_features(article_data)

        assert features["inbound_links"] == 0.0
        assert features["outbound_links"] == 0.0
        assert features["total_links"] == 0.0
        assert features["link_density"] == 0.0
        assert features["connectivity_score"] == 0.0


class TestFeatureIntegration:
    """Integration tests for feature extraction functions."""

    def test_albert_einstein_features(self) -> None:
        """Test feature extraction with Albert Einstein article data."""
        # Mock comprehensive Albert Einstein article data
        article_data: Dict[str, Any] = {
            "data": {
                "parse": {
                    "sections": [
                        {"index": 1, "line": "Introduction", "level": 1},
                        {"index": 2, "line": "Early life and education", "level": 2},
                        {"index": 3, "line": "Academic career", "level": 2},
                        {"index": 4, "line": "Personal life", "level": 2},
                        {"index": 5, "line": "Scientific career", "level": 1},
                        {"index": 6, "line": "Annus mirabilis papers", "level": 2},
                        {"index": 7, "line": "General relativity", "level": 2},
                        {"index": 8, "line": "Later years", "level": 1},
                        {"index": 9, "line": "Death", "level": 2},
                        {"index": 10, "line": "Legacy", "level": 1},
                        {"index": 11, "line": "Awards and honors", "level": 2},
                        {"index": 12, "line": "References", "level": 1},
                        {"index": 13, "line": "External links", "level": 1},
                    ],
                    "text": {
                        "*": "Albert Einstein was a German-born theoretical physicist. "
                        * 200
                    },
                },
                "query": {
                    "pages": {
                        "123": {
                            "templates": [
                                {"title": "Template:Infobox scientist"},
                                {"title": "Template:Authority control"},
                                {"title": "Template:Navbox"},
                            ],
                            "extlinks": [
                                {"url": "https://harvard.edu/einstein"},
                                {"url": "https://mit.edu/physics"},
                                {"url": "https://bbc.com/science"},
                                {"url": "https://nasa.gov/relativity"},
                                {"url": "https://stanford.edu/research"},
                                {"url": "https://nature.com/physics"},
                                {"url": "https://science.org/einstein"},
                                {"url": "https://princeton.edu/physics"},
                            ],
                            "revisions": [
                                {
                                    "revid": 1,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=1)
                                    ).isoformat(),
                                    "user": "PhysicsEditor",
                                    "comment": "Updated references",
                                    "size": 5000,
                                },
                                {
                                    "revid": 2,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=5)
                                    ).isoformat(),
                                    "user": "ScienceBot",
                                    "comment": "Bot edit",
                                    "size": 4800,
                                },
                                {
                                    "revid": 3,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=10)
                                    ).isoformat(),
                                    "user": "EinsteinFan",
                                    "comment": "Added new section",
                                    "size": 5200,
                                },
                                {
                                    "revid": 4,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=20)
                                    ).isoformat(),
                                    "user": "PhysicsEditor",
                                    "comment": "Minor corrections",
                                    "size": 5100,
                                },
                                {
                                    "revid": 5,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=50)
                                    ).isoformat(),
                                    "user": "AcademicEditor",
                                    "comment": "Updated bibliography",
                                    "size": 4900,
                                },
                            ],
                            "links": [
                                {"title": "Physics", "ns": 0},
                                {"title": "Theory of relativity", "ns": 0},
                                {"title": "Quantum mechanics", "ns": 0},
                                {"title": "Nobel Prize", "ns": 0},
                                {"title": "Princeton University", "ns": 0},
                            ],
                        }
                    },
                    "backlinks": [
                        {"title": "Physics", "ns": 0},
                        {"title": "Theory of relativity", "ns": 0},
                        {"title": "Quantum mechanics", "ns": 0},
                        {"title": "Nobel Prize in Physics", "ns": 0},
                        {"title": "Princeton University", "ns": 0},
                        {"title": "Swiss Federal Polytechnic", "ns": 0},
                        {"title": "Mass-energy equivalence", "ns": 0},
                        {"title": "Photoelectric effect", "ns": 0},
                        {"title": "Brownian motion", "ns": 0},
                        {"title": "Special relativity", "ns": 0},
                    ],
                },
            }
        }

        # Extract all features
        structure_feats = structure_features(article_data)
        sourcing_feats = sourcing_features(article_data)
        editorial_feats = editorial_features(article_data)
        network_feats = network_features(article_data)

        # Combine all features
        all_features = {
            **structure_feats,
            **sourcing_feats,
            **editorial_feats,
            **network_feats,
        }

        # Verify we have more than 20 features
        assert len(all_features) > 20, f"Expected >20 features, got {len(all_features)}"

        # Verify key features are present and reasonable
        assert structure_feats["section_count"] == 13.0
        assert structure_feats["template_count"] == 3.0
        assert structure_feats["has_infobox"] == 1.0

        assert sourcing_feats["citation_count"] == 8.0
        assert sourcing_feats["academic_source_ratio"] >= 0.5  # Multiple .edu domains

        assert (
            editorial_feats["total_editors"] == 4.0
        )  # PhysicsEditor, ScienceBot, EinsteinFan, AcademicEditor
        assert editorial_feats["total_revisions"] == 5.0
        assert editorial_feats["bot_edit_ratio"] == 0.2  # 1 out of 5 edits

        assert network_feats["inbound_links"] == 10.0
        assert network_feats["outbound_links"] == 5.0

        # Verify all features are numeric
        for feature_name, feature_value in all_features.items():
            assert isinstance(
                feature_value, (int, float)
            ), f"Feature {feature_name} is not numeric: {type(feature_value)}"
            assert not math.isnan(feature_value), f"Feature {feature_name} is NaN"
            assert not math.isinf(feature_value), f"Feature {feature_name} is infinite"

    def test_dog_features(self) -> None:
        """Test feature extraction with Dog article data."""
        # Mock comprehensive Dog article data
        article_data: Dict[str, Any] = {
            "data": {
                "parse": {
                    "sections": [
                        {"index": 1, "line": "Introduction", "level": 1},
                        {"index": 2, "line": "Taxonomy", "level": 2},
                        {"index": 3, "line": "Evolution", "level": 2},
                        {"index": 4, "line": "Domestication", "level": 2},
                        {"index": 5, "line": "Physical characteristics", "level": 1},
                        {"index": 6, "line": "Size and weight", "level": 2},
                        {"index": 7, "line": "Coat", "level": 2},
                        {"index": 8, "line": "Behavior", "level": 1},
                        {"index": 9, "line": "Intelligence", "level": 2},
                        {"index": 10, "line": "Communication", "level": 2},
                        {"index": 11, "line": "Breeds", "level": 1},
                        {"index": 12, "line": "Health", "level": 1},
                        {"index": 13, "line": "References", "level": 1},
                    ],
                    "text": {
                        "*": "The dog is a domesticated descendant of the wolf. " * 150
                    },
                },
                "query": {
                    "pages": {
                        "456": {
                            "templates": [
                                {"title": "Template:Infobox animal"},
                                {"title": "Template:Taxonbar"},
                                {"title": "Template:Authority control"},
                            ],
                            "extlinks": [
                                {"url": "https://akc.org/breeds"},
                                {"url": "https://fci.be/dogs"},
                                {"url": "https://vetmed.ucdavis.edu"},
                                {"url": "https://nationalgeographic.com/animals"},
                                {"url": "https://sciencedirect.com/canine"},
                            ],
                            "revisions": [
                                {
                                    "revid": 1,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=2)
                                    ).isoformat(),
                                    "user": "DogLover",
                                    "comment": "Updated breed information",
                                    "size": 3000,
                                },
                                {
                                    "revid": 2,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=7)
                                    ).isoformat(),
                                    "user": "VetEditor",
                                    "comment": "Added health section",
                                    "size": 3200,
                                },
                                {
                                    "revid": 3,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=15)
                                    ).isoformat(),
                                    "user": "AnimalBot",
                                    "comment": "Bot maintenance",
                                    "size": 3100,
                                },
                                {
                                    "revid": 4,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=30)
                                    ).isoformat(),
                                    "user": "DogLover",
                                    "comment": "Expanded behavior section",
                                    "size": 3300,
                                },
                                {
                                    "revid": 5,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=60)
                                    ).isoformat(),
                                    "user": "IP:192.168.1.1",
                                    "comment": "Anonymous edit",
                                    "size": 2900,
                                },
                                {
                                    "revid": 6,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=120)
                                    ).isoformat(),
                                    "user": "AnimalExpert",
                                    "comment": "Updated taxonomy",
                                    "size": 3400,
                                },
                            ],
                            "links": [
                                {"title": "Wolf", "ns": 0},
                                {"title": "Domestication", "ns": 0},
                                {"title": "Canine", "ns": 0},
                                {"title": "Pet", "ns": 0},
                            ],
                        }
                    },
                    "backlinks": [
                        {"title": "Wolf", "ns": 0},
                        {"title": "Domestication", "ns": 0},
                        {"title": "Pet", "ns": 0},
                        {"title": "Canine", "ns": 0},
                        {"title": "Animal", "ns": 0},
                        {"title": "Mammal", "ns": 0},
                        {"title": "Carnivore", "ns": 0},
                    ],
                },
            }
        }

        # Extract all features
        structure_feats = structure_features(article_data)
        sourcing_feats = sourcing_features(article_data)
        editorial_feats = editorial_features(article_data)
        network_feats = network_features(article_data)

        # Combine all features
        all_features = {
            **structure_feats,
            **sourcing_feats,
            **editorial_feats,
            **network_feats,
        }

        # Verify we have more than 20 features
        assert len(all_features) > 20, f"Expected >20 features, got {len(all_features)}"

        # Verify key features are present and reasonable
        assert structure_feats["section_count"] == 13.0
        assert structure_feats["template_count"] == 3.0
        assert structure_feats["has_infobox"] == 1.0  # Has infobox animal template
        assert structure_feats["has_taxonbar"] == 1.0  # Has taxonbar template

        assert sourcing_feats["citation_count"] == 5.0
        assert sourcing_feats["org_source_ratio"] == 0.2  # Only akc.org is .org domain

        assert (
            editorial_feats["total_editors"] == 5.0
        )  # DogLover, VetEditor, AnimalBot, IP user, AnimalExpert
        assert editorial_feats["total_revisions"] == 6.0
        assert editorial_feats["bot_edit_ratio"] == 1 / 6  # 1 out of 6 edits
        assert editorial_feats["anonymous_edit_ratio"] == 1 / 6  # 1 out of 6 edits

        assert network_feats["inbound_links"] == 7.0
        assert network_feats["outbound_links"] == 4.0

        # Verify all features are numeric
        for feature_name, feature_value in all_features.items():
            assert isinstance(
                feature_value, (int, float)
            ), f"Feature {feature_name} is not numeric: {type(feature_value)}"
            assert not math.isnan(feature_value), f"Feature {feature_name} is NaN"
            assert not math.isinf(feature_value), f"Feature {feature_name} is infinite"

    def test_feature_normalization(self) -> None:
        """Test that features are properly normalized."""
        # Test with minimal data
        minimal_data: Dict[str, Any] = {
            "data": {
                "parse": {
                    "sections": [{"index": 1, "line": "Test", "level": 1}],
                    "text": {"*": "Short content"},
                }
            }
        }

        # Test with extensive data
        extensive_data: Dict[str, Any] = {
            "data": {
                "parse": {
                    "sections": [
                        {"index": i, "line": f"Section {i}", "level": 1}
                        for i in range(1, 21)
                    ],
                    "text": {"*": "Very long content " * 1000},
                },
                "query": {
                    "pages": {
                        "123": {
                            "extlinks": [
                                {"url": f"https://example{i}.com"} for i in range(50)
                            ],
                            "revisions": [
                                {
                                    "revid": i,
                                    "timestamp": (
                                        datetime.now(timezone.utc) - timedelta(days=i)
                                    ).isoformat(),
                                    "user": f"Editor{i}",
                                    "comment": f"Edit {i}",
                                    "size": 1000 + i,
                                }
                                for i in range(1, 21)
                            ],
                        }
                    },
                    "backlinks": [{"title": f"Page {i}", "ns": 0} for i in range(30)],
                },
            }
        }

        # Extract features from both
        minimal_features = structure_features(minimal_data)
        extensive_features = structure_features(extensive_data)

        # Verify normalization works (ratios should be reasonable)
        assert minimal_features["sections_per_1k_chars"] > 0
        assert extensive_features["sections_per_1k_chars"] > 0

        # Extensive data should have more sections but normalized metrics should be reasonable
        assert extensive_features["section_count"] > minimal_features["section_count"]
        assert (
            extensive_features["log_section_count"]
            > minimal_features["log_section_count"]
        )
