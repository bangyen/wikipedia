"""Unit tests for WikiClient functionality."""

import json
import math
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
import requests

from src.ingest.wiki_client import WikiClient


class TestWikiClient:
    """Test cases for WikiClient class."""
    
    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.client = WikiClient(
            cache_ttl=60,
            max_cache_size=100,
            rate_limit_delay=0.01,
            max_retries=2,
        )
    
    def test_init(self) -> None:
        """Test WikiClient initialization."""
        client = WikiClient()
        assert client.base_url == "https://en.wikipedia.org/w/api.php"
        assert client.pageviews_url == "https://wikimedia.org/api/rest_v1/metrics/pageviews"
        assert client.rate_limit_delay == 0.1
        assert client.max_retries == 3
        assert len(client._cache) == 0
    
    def test_init_custom_params(self) -> None:
        """Test WikiClient initialization with custom parameters."""
        client = WikiClient(
            base_url="https://test.wikipedia.org/w/api.php",
            pageviews_url="https://test.wikimedia.org/api/rest_v1/metrics/pageviews",
            cache_ttl=1800,
            max_cache_size=500,
            rate_limit_delay=0.5,
            max_retries=5,
        )
        assert client.base_url == "https://test.wikipedia.org/w/api.php"
        assert client.pageviews_url == "https://test.wikimedia.org/api/rest_v1/metrics/pageviews"
        assert client.rate_limit_delay == 0.5
        assert client.max_retries == 5
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_get_page_content(self, mock_get: Mock) -> None:
        """Test get_page_content method."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "title": "Albert Einstein",
                        "extract": "Albert Einstein was a German-born theoretical physicist..."
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_page_content("Albert Einstein")
        
        # Verify response structure
        assert "timestamp" in result
        assert "title" in result
        assert "data" in result
        assert result["title"] == "Albert Einstein"
        
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
        
        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "Albert Einstein" in str(call_args[1]['params']['titles'])
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_get_sections(self, mock_get: Mock) -> None:
        """Test get_sections method."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "parse": {
                "sections": [
                    {"index": 1, "line": "Early life", "anchor": "Early_life"},
                    {"index": 2, "line": "Career", "anchor": "Career"},
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_sections("Albert Einstein")
        
        assert "timestamp" in result
        assert "title" in result
        assert "data" in result
        assert result["title"] == "Albert Einstein"
        
        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "Albert Einstein" in str(call_args[1]['params']['page'])
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_get_templates(self, mock_get: Mock) -> None:
        """Test get_templates method."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "templates": [
                            {"title": "Template:Infobox scientist"},
                            {"title": "Template:Authority control"},
                        ]
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_templates("Albert Einstein")
        
        assert "timestamp" in result
        assert "title" in result
        assert "data" in result
        assert result["title"] == "Albert Einstein"
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_get_revisions(self, mock_get: Mock) -> None:
        """Test get_revisions method."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "revisions": [
                            {
                                "revid": 123456,
                                "timestamp": "2023-01-01T00:00:00Z",
                                "user": "TestUser",
                                "comment": "Test edit",
                                "size": 5000,
                            }
                        ]
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_revisions("Albert Einstein", rvlimit=5)
        
        assert "timestamp" in result
        assert "title" in result
        assert "data" in result
        assert result["title"] == "Albert Einstein"
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_get_backlinks(self, mock_get: Mock) -> None:
        """Test get_backlinks method."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": {
                "backlinks": [
                    {"title": "Physics", "ns": 0},
                    {"title": "Theory of relativity", "ns": 0},
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_backlinks("Albert Einstein", bllimit=20)
        
        assert "timestamp" in result
        assert "title" in result
        assert "data" in result
        assert result["title"] == "Albert Einstein"
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_get_pageviews(self, mock_get: Mock) -> None:
        """Test get_pageviews method."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "project": "en.wikipedia",
                    "article": "Albert_Einstein",
                    "granularity": "daily",
                    "timestamp": "2023010100",
                    "access": "all-access",
                    "agent": "user",
                    "views": 1500,
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_pageviews(
            "Albert Einstein",
            start_date="20230101",
            end_date="20230107",
        )
        
        assert "timestamp" in result
        assert "title" in result
        assert "data" in result
        assert result["title"] == "Albert Einstein"
        assert result["start_date"] == "20230101"
        assert result["end_date"] == "20230107"
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_get_citations(self, mock_get: Mock) -> None:
        """Test get_citations method."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "extlinks": [
                            {"url": "https://example.com/source1"},
                            {"url": "https://example.com/source2"},
                        ]
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_citations("Albert Einstein", ellimit=30)
        
        assert "timestamp" in result
        assert "title" in result
        assert "data" in result
        assert result["title"] == "Albert Einstein"
    
    def test_cache_functionality(self) -> None:
        """Test caching functionality."""
        # Clear cache
        self.client.clear_cache()
        assert len(self.client._cache) == 0
        
        # Get cache info
        cache_info = self.client.get_cache_info()
        assert "cache_size" in cache_info
        assert "cache_maxsize" in cache_info
        assert "cache_ttl" in cache_info
        assert cache_info["cache_size"] == 0
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_caching_works(self, mock_get: Mock) -> None:
        """Test that caching prevents duplicate API calls."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # First call
        self.client.get_page_content("Test Page")
        assert mock_get.call_count == 1
        
        # Second call should use cache
        self.client.get_page_content("Test Page")
        assert mock_get.call_count == 1  # Should not increase
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_rate_limiting(self, mock_get: Mock) -> None:
        """Test rate limiting functionality."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock time.sleep to avoid actual delays in tests
        with patch('src.ingest.wiki_client.time.sleep') as mock_sleep:
            self.client.get_page_content("Page 1")
            self.client.get_page_content("Page 2")
            
            # Should have called sleep for rate limiting
            assert mock_sleep.call_count >= 1
    
    @patch('src.ingest.wiki_client.requests.Session.get')
    def test_retry_logic(self, mock_get: Mock) -> None:
        """Test retry logic for failed requests."""
        # Mock response that fails first, then succeeds
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.side_effect = [
            requests.RequestException("Network error"),
            None,  # Success on second attempt
        ]
        mock_get.return_value = mock_response
        
        # Should retry and eventually succeed
        result = self.client.get_page_content("Test Page")
        assert "timestamp" in result
        assert mock_get.call_count == 2  # Should have retried once
    
    def test_cache_key_generation(self) -> None:
        """Test cache key generation is consistent."""
        key1 = self.client._get_cache_key("test_method", param1="value1", param2="value2")
        key2 = self.client._get_cache_key("test_method", param2="value2", param1="value1")
        
        # Same parameters in different order should generate same key
        assert key1 == key2
        
        # Different parameters should generate different keys
        key3 = self.client._get_cache_key("test_method", param1="different")
        assert key1 != key3
    
    def test_json_serializable_output(self) -> None:
        """Test that all methods return JSON-serializable output."""
        with patch('src.ingest.wiki_client.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"test": "data"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            methods = [
                ("get_page_content", ["Test Page"]),
                ("get_sections", ["Test Page"]),
                ("get_templates", ["Test Page"]),
                ("get_revisions", ["Test Page"]),
                ("get_backlinks", ["Test Page"]),
                ("get_pageviews", ["Test Page", "20230101", "20230107"]),
                ("get_citations", ["Test Page"]),
            ]
            
            for method_name, args in methods:
                method = getattr(self.client, method_name)
                result = method(*args)
                
                # Should be JSON serializable
                json.dumps(result)
                
                # Should contain required fields
                assert "timestamp" in result
                assert "data" in result
                
                # Timestamp should be valid ISO format
                datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
