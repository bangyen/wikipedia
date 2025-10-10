"""Unit tests for Wikidata client and feature extraction.

This module tests the WikidataClient functionality including API calls,
caching, error handling, and feature extraction from Wikidata data.
"""

import pytest
from unittest.mock import Mock, patch

from features.wikidata import (
    WikidataClient,
    wikidata_features,
    _get_zero_wikidata_features,
)


class TestWikidataClient:
    """Test cases for WikidataClient class."""

    def test_init(self) -> None:
        """Test WikidataClient initialization."""
        client = WikidataClient()
        assert client.base_url == "https://www.wikidata.org/w/api.php"
        assert client.rate_limit_delay == 0.1
        assert client.max_retries == 3
        assert client._cache.maxsize == 1000
        assert client._cache.ttl == 3600

    def test_init_custom_params(self) -> None:
        """Test WikidataClient initialization with custom parameters."""
        client = WikidataClient(
            base_url="https://test.wikidata.org/w/api.php",
            cache_ttl=1800,
            max_cache_size=500,
            rate_limit_delay=0.2,
            max_retries=5,
        )
        assert client.base_url == "https://test.wikidata.org/w/api.php"
        assert client.rate_limit_delay == 0.2
        assert client.max_retries == 5
        assert client._cache.maxsize == 500
        assert client._cache.ttl == 1800

    @patch("features.wikidata.requests.Session.get")
    def test_get_wikidata_id_success(self, mock_get: Mock) -> None:
        """Test successful Wikidata ID retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"entities": {"Q123": {"id": "Q123"}}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = WikidataClient()
        result = client.get_wikidata_id("Albert Einstein")

        assert result == "Q123"
        mock_get.assert_called_once()

    @patch("features.wikidata.requests.Session.get")
    def test_get_wikidata_id_not_found(self, mock_get: Mock) -> None:
        """Test Wikidata ID retrieval when not found."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "entities": {"Q123": {"id": "Q123", "missing": "entity"}}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = WikidataClient()
        result = client.get_wikidata_id("NonExistentArticle")

        assert result is None

    @patch("features.wikidata.requests.Session.get")
    def test_get_wikidata_id_error(self, mock_get: Mock) -> None:
        """Test Wikidata ID retrieval with API error."""
        mock_get.side_effect = Exception("API Error")

        client = WikidataClient()
        result = client.get_wikidata_id("Albert Einstein")

        assert result is None

    @patch("features.wikidata.requests.Session.get")
    def test_get_statements_count_success(self, mock_get: Mock) -> None:
        """Test successful statements count retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "entities": {
                "Q123": {
                    "claims": {
                        "P31": [
                            {"id": "Q123$123", "references": [{"snaks": {}}]},
                            {"id": "Q123$124", "references": []},
                        ],
                        "P21": [
                            {"id": "Q123$125", "references": [{"snaks": {}}]},
                        ],
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = WikidataClient()
        result = client.get_statements_count("Q123")

        assert result["total_statements"] == 3
        assert result["referenced_statements"] == 2

    @patch("features.wikidata.requests.Session.get")
    def test_get_statements_count_error(self, mock_get: Mock) -> None:
        """Test statements count retrieval with API error."""
        mock_get.side_effect = Exception("API Error")

        client = WikidataClient()
        result = client.get_statements_count("Q123")

        assert result["total_statements"] == 0
        assert result["referenced_statements"] == 0

    @patch("features.wikidata.requests.Session.get")
    def test_get_sitelinks_count_success(self, mock_get: Mock) -> None:
        """Test successful sitelinks count retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "entities": {
                "Q123": {
                    "sitelinks": {
                        "enwiki": {"title": "Albert Einstein"},
                        "dewiki": {"title": "Albert Einstein"},
                        "frwiki": {"title": "Albert Einstein"},
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = WikidataClient()
        result = client.get_sitelinks_count("Q123")

        assert result == 3

    @patch("features.wikidata.requests.Session.get")
    def test_get_sitelinks_count_error(self, mock_get: Mock) -> None:
        """Test sitelinks count retrieval with API error."""
        mock_get.side_effect = Exception("API Error")

        client = WikidataClient()
        result = client.get_sitelinks_count("Q123")

        assert result == 0

    @patch.object(WikidataClient, "get_wikidata_id")
    @patch.object(WikidataClient, "get_statements_count")
    @patch.object(WikidataClient, "get_sitelinks_count")
    def test_get_completeness_data_success(
        self, mock_sitelinks: Mock, mock_statements: Mock, mock_wikidata_id: Mock
    ) -> None:
        """Test successful completeness data retrieval."""
        # Mock responses
        mock_wikidata_id.return_value = "Q123"
        mock_statements.return_value = {
            "total_statements": 100,
            "referenced_statements": 50,
        }
        mock_sitelinks.return_value = 25

        client = WikidataClient()
        result = client.get_completeness_data("Albert Einstein")

        assert result["wikidata_id"] == "Q123"
        assert result["total_statements"] == 100
        assert result["referenced_statements"] == 50
        assert result["sitelinks_count"] == 25
        assert result["claim_density"] == 4.0  # 100/25
        assert result["referenced_ratio"] == 0.5  # 50/100
        assert result["completeness_score"] > 0
        assert "timestamp" in result

    @patch.object(WikidataClient, "get_wikidata_id")
    def test_get_completeness_data_no_wikidata(self, mock_wikidata_id: Mock) -> None:
        """Test completeness data retrieval when no Wikidata ID found."""
        mock_wikidata_id.return_value = None

        client = WikidataClient()
        result = client.get_completeness_data("NonExistentArticle")

        assert result["wikidata_id"] is None
        assert result["total_statements"] == 0
        assert result["referenced_statements"] == 0
        assert result["sitelinks_count"] == 0
        assert result["claim_density"] == 0.0
        assert result["referenced_ratio"] == 0.0
        assert result["completeness_score"] == 0.0

    def test_clear_cache(self) -> None:
        """Test cache clearing functionality."""
        client = WikidataClient()

        # Add something to cache
        client._cache["test_key"] = "test_value"
        assert len(client._cache) == 1

        # Clear cache
        client.clear_cache()
        assert len(client._cache) == 0

    def test_get_cache_info(self) -> None:
        """Test cache info retrieval."""
        client = WikidataClient()
        info = client.get_cache_info()

        assert "cache_size" in info
        assert "cache_maxsize" in info
        assert "cache_ttl" in info
        assert info["cache_size"] == 0
        assert info["cache_maxsize"] == 1000
        assert info["cache_ttl"] == 3600


class TestWikidataFeatures:
    """Test cases for wikidata_features function."""

    def test_wikidata_features_success(self) -> None:
        """Test successful Wikidata feature extraction."""
        article_data = {"title": "Albert Einstein", "data": {}}

        with patch("features.wikidata.WikidataClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_completeness_data.return_value = {
                "wikidata_id": "Q123",
                "total_statements": 100,
                "referenced_statements": 50,
                "sitelinks_count": 25,
                "claim_density": 4.0,
                "referenced_ratio": 0.5,
                "completeness_score": 0.4,
            }

            features = wikidata_features(article_data)

            assert features["wikidata_statements"] == 100.0
            assert features["wikidata_referenced_statements"] == 50.0
            assert features["wikidata_sitelinks"] == 25.0
            assert features["wikidata_claim_density"] == 4.0
            assert features["wikidata_referenced_ratio"] == 0.5
            assert features["wikidata_completeness_score"] == 0.4
            assert features["wikidata_has_data"] == 1.0
            assert features["log_wikidata_statements"] > 0
            assert features["log_wikidata_sitelinks"] > 0
            assert features["wikidata_statement_quality"] == 0.5
            assert features["wikidata_connectivity"] > 0
            assert features["wikidata_factual_richness"] > 0

    def test_wikidata_features_no_title(self) -> None:
        """Test Wikidata feature extraction with no title."""
        article_data: dict = {"data": {}}

        features = wikidata_features(article_data)

        expected = _get_zero_wikidata_features()
        assert features == expected

    def test_wikidata_features_empty_title(self) -> None:
        """Test Wikidata feature extraction with empty title."""
        article_data = {"title": "", "data": {}}

        features = wikidata_features(article_data)

        expected = _get_zero_wikidata_features()
        assert features == expected

    def test_wikidata_features_error(self) -> None:
        """Test Wikidata feature extraction with error."""
        article_data = {"title": "Albert Einstein", "data": {}}

        with patch("features.wikidata.WikidataClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_completeness_data.side_effect = Exception("API Error")

            features = wikidata_features(article_data)

            expected = _get_zero_wikidata_features()
            assert features == expected


class TestHelperFunctions:
    """Test cases for helper functions."""

    def test_get_zero_wikidata_features(self) -> None:
        """Test zero Wikidata features function."""
        features = _get_zero_wikidata_features()

        assert isinstance(features, dict)
        assert len(features) == 12

        # Check all expected keys are present
        expected_keys = [
            "wikidata_statements",
            "wikidata_referenced_statements",
            "wikidata_sitelinks",
            "wikidata_claim_density",
            "wikidata_referenced_ratio",
            "wikidata_completeness_score",
            "wikidata_has_data",
            "log_wikidata_statements",
            "log_wikidata_sitelinks",
            "wikidata_statement_quality",
            "wikidata_connectivity",
            "wikidata_factual_richness",
        ]

        for key in expected_keys:
            assert key in features
            assert features[key] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
