"""Wikidata API client for fetching factual completeness signals.

This module provides a WikidataClient that fetches Wikipedia article metadata
from Wikidata, including statements, references, and sitelinks. It computes
completeness metrics like claim density and referenced ratio to assess
the factual completeness of Wikipedia articles.
"""

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests  # type: ignore
from cachetools import TTLCache  # type: ignore
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class WikidataClient:
    """Client for Wikidata API with caching and retry support.

    This client fetches factual completeness signals from Wikidata including
    the number of statements, referenced statements, and sitelinks for Wikipedia
    articles. It computes completeness metrics to assess article quality.
    """

    def __init__(
        self,
        base_url: str = "https://www.wikidata.org/w/api.php",
        cache_ttl: int = 3600,
        max_cache_size: int = 1000,
        rate_limit_delay: float = 0.1,
        max_retries: int = 3,
    ) -> None:
        """Initialize the WikidataClient with configuration parameters.

        Args:
            base_url: Base URL for Wikidata API
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum number of cached responses
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

        # Initialize cache with TTL
        self._cache = TTLCache(maxsize=max_cache_size, ttl=cache_ttl)

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "WikidataClient/1.0 (https://github.com/yourusername/wikipedia; contact@example.com)"
            }
        )

        # Rate limiting
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)

        self._last_request_time = time.time()

    def _make_request(
        self,
        url: str,
        params: Dict[str, Any],
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make a request with caching and retry logic.

        Args:
            url: Request URL
            params: Request parameters
            cache_key: Optional cache key for response caching

        Returns:
            JSON response as dictionary

        Raises:
            requests.RequestException: If request fails after retries
        """
        # Check cache first
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(requests.RequestException),
        )
        def _request() -> Dict[str, Any]:
            self._rate_limit()
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()  # type: ignore

        result = _request()

        # Cache the result
        if cache_key:
            self._cache[cache_key] = result

        return result  # type: ignore

    def _get_cache_key(self, method: str, **kwargs: Any) -> str:
        """Generate a cache key for the given method and parameters."""
        # Sort kwargs for consistent cache keys
        sorted_kwargs = sorted(kwargs.items())
        key_data = f"{method}:{sorted_kwargs}"
        return str(hash(key_data))

    def get_wikidata_id(
        self, wikipedia_title: str, wikipedia_lang: str = "en"
    ) -> Optional[str]:
        """Get Wikidata ID for a Wikipedia article.

        Args:
            wikipedia_title: Wikipedia article title
            wikipedia_lang: Wikipedia language code (default: en)

        Returns:
            Wikidata ID (Q-number) if found, None otherwise
        """
        cache_key = self._get_cache_key(
            "wikidata_id", title=wikipedia_title, lang=wikipedia_lang
        )

        params = {
            "format": "json",
            "action": "wbgetentities",
            "sites": f"{wikipedia_lang}wiki",
            "titles": wikipedia_title,
            "props": "info",
        }

        try:
            response = self._make_request(self.base_url, params, cache_key)

            # Extract Wikidata ID from response
            entities = response.get("entities", {})
            for entity_id, entity_data in entities.items():
                if entity_id.startswith("Q") and "missing" not in entity_data:
                    return str(entity_id)

            return None
        except Exception:
            return None

    def get_statements_count(self, wikidata_id: str) -> Dict[str, int]:
        """Get statement counts for a Wikidata entity.

        Args:
            wikidata_id: Wikidata entity ID (Q-number)

        Returns:
            Dictionary containing statement counts
        """
        cache_key = self._get_cache_key("statements", id=wikidata_id)

        params = {
            "format": "json",
            "action": "wbgetentities",
            "ids": wikidata_id,
            "props": "claims",
        }

        try:
            response = self._make_request(self.base_url, params, cache_key)

            entities = response.get("entities", {})
            entity_data = entities.get(wikidata_id, {})
            claims = entity_data.get("claims", {})

            total_statements = 0
            referenced_statements = 0

            for property_id, statement_list in claims.items():
                if isinstance(statement_list, list):
                    total_statements += len(statement_list)

                    # Count referenced statements
                    for statement in statement_list:
                        references = statement.get("references", [])
                        if references:
                            referenced_statements += 1

            return {
                "total_statements": total_statements,
                "referenced_statements": referenced_statements,
            }
        except Exception:
            return {"total_statements": 0, "referenced_statements": 0}

    def get_sitelinks_count(self, wikidata_id: str) -> int:
        """Get sitelinks count for a Wikidata entity.

        Args:
            wikidata_id: Wikidata entity ID (Q-number)

        Returns:
            Number of sitelinks
        """
        cache_key = self._get_cache_key("sitelinks", id=wikidata_id)

        params = {
            "format": "json",
            "action": "wbgetentities",
            "ids": wikidata_id,
            "props": "sitelinks",
        }

        try:
            response = self._make_request(self.base_url, params, cache_key)

            entities = response.get("entities", {})
            entity_data = entities.get(wikidata_id, {})
            sitelinks = entity_data.get("sitelinks", {})

            return len(sitelinks)
        except Exception:
            return 0

    def get_completeness_data(
        self, wikipedia_title: str, wikipedia_lang: str = "en"
    ) -> Dict[str, Any]:
        """Get complete Wikidata data for a Wikipedia article.

        Args:
            wikipedia_title: Wikipedia article title
            wikipedia_lang: Wikipedia language code (default: en)

        Returns:
            Dictionary containing all Wikidata completeness data
        """
        # Get Wikidata ID
        wikidata_id = self.get_wikidata_id(wikipedia_title, wikipedia_lang)

        if not wikidata_id:
            return {
                "wikidata_id": None,
                "total_statements": 0,
                "referenced_statements": 0,
                "sitelinks_count": 0,
                "claim_density": 0.0,
                "referenced_ratio": 0.0,
                "completeness_score": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Get statement counts
        statement_counts = self.get_statements_count(wikidata_id)

        # Get sitelinks count
        sitelinks_count = self.get_sitelinks_count(wikidata_id)

        # Compute completeness metrics
        total_statements = statement_counts["total_statements"]
        referenced_statements = statement_counts["referenced_statements"]

        # Claim density (statements per sitelink)
        claim_density = (
            total_statements / sitelinks_count if sitelinks_count > 0 else 0.0
        )

        # Referenced ratio (proportion of statements with references)
        referenced_ratio = (
            referenced_statements / total_statements if total_statements > 0 else 0.0
        )

        # Overall completeness score (weighted combination)
        completeness_score = (
            min(claim_density / 10.0, 1.0) * 0.4  # Normalize claim density
            + referenced_ratio * 0.6  # Weight referenced ratio more heavily
        )

        return {
            "wikidata_id": wikidata_id,
            "total_statements": total_statements,
            "referenced_statements": referenced_statements,
            "sitelinks_count": sitelinks_count,
            "claim_density": claim_density,
            "referenced_ratio": referenced_ratio,
            "completeness_score": completeness_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state.

        Returns:
            Dictionary containing cache statistics
        """
        return {
            "cache_size": len(self._cache),
            "cache_maxsize": self._cache.maxsize,
            "cache_ttl": self._cache.ttl,
        }


def wikidata_features(article_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract Wikidata completeness features from Wikipedia article data.

    Analyzes Wikidata metadata to compute factual completeness signals including
    statement counts, reference ratios, and sitelink connectivity metrics.

    Args:
        article_data: Raw Wikipedia article JSON data containing title and metadata

    Returns:
        Dictionary of normalized Wikidata completeness features including:
        - wikidata_statements: Total number of Wikidata statements
        - wikidata_referenced_statements: Number of referenced statements
        - wikidata_sitelinks: Number of sitelinks
        - wikidata_claim_density: Claims per sitelink ratio
        - wikidata_referenced_ratio: Proportion of statements with references
        - wikidata_completeness_score: Overall completeness score
        - wikidata_has_data: Whether Wikidata data is available
        - log_wikidata_statements: Log-scaled statement count
        - log_wikidata_sitelinks: Log-scaled sitelinks count
    """
    features = {}

    # Extract title from article data
    title = article_data.get("title", "")
    if not title:
        return _get_zero_wikidata_features()

    # Initialize Wikidata client
    client = WikidataClient()

    try:
        # Get completeness data
        completeness_data = client.get_completeness_data(title)

        # Extract features
        features["wikidata_statements"] = float(completeness_data["total_statements"])
        features["wikidata_referenced_statements"] = float(
            completeness_data["referenced_statements"]
        )
        features["wikidata_sitelinks"] = float(completeness_data["sitelinks_count"])
        features["wikidata_claim_density"] = completeness_data["claim_density"]
        features["wikidata_referenced_ratio"] = completeness_data["referenced_ratio"]
        features["wikidata_completeness_score"] = completeness_data[
            "completeness_score"
        ]
        features["wikidata_has_data"] = float(
            completeness_data["wikidata_id"] is not None
        )

        # Log scaling for large values
        features["log_wikidata_statements"] = math.log(
            max(features["wikidata_statements"], 1)
        )
        features["log_wikidata_sitelinks"] = math.log(
            max(features["wikidata_sitelinks"], 1)
        )

        # Additional derived features
        if features["wikidata_statements"] > 0:
            features["wikidata_statement_quality"] = (
                features["wikidata_referenced_statements"]
                / features["wikidata_statements"]
            )
        else:
            features["wikidata_statement_quality"] = 0.0

        # Connectivity score (based on sitelinks)
        if features["wikidata_sitelinks"] > 0:
            features["wikidata_connectivity"] = min(
                features["wikidata_sitelinks"] / 100.0, 1.0
            )
        else:
            features["wikidata_connectivity"] = 0.0

        # Factual richness score (combination of statements and references)
        if features["wikidata_statements"] > 0:
            features["wikidata_factual_richness"] = (
                features["wikidata_statements"] * features["wikidata_referenced_ratio"]
            ) / 100.0  # Normalize
        else:
            features["wikidata_factual_richness"] = 0.0

    except Exception:
        # Return zero values if Wikidata data is unavailable
        return _get_zero_wikidata_features()

    return features


def _get_zero_wikidata_features() -> Dict[str, float]:
    """Return zero values for all Wikidata features."""
    return {
        "wikidata_statements": 0.0,
        "wikidata_referenced_statements": 0.0,
        "wikidata_sitelinks": 0.0,
        "wikidata_claim_density": 0.0,
        "wikidata_referenced_ratio": 0.0,
        "wikidata_completeness_score": 0.0,
        "wikidata_has_data": 0.0,
        "log_wikidata_statements": 0.0,
        "log_wikidata_sitelinks": 0.0,
        "wikidata_statement_quality": 0.0,
        "wikidata_connectivity": 0.0,
        "wikidata_factual_richness": 0.0,
    }
