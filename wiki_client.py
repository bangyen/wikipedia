"""Wikipedia API client with caching, rate limiting, and retry logic.

This module provides a comprehensive client for interacting with Wikipedia's
MediaWiki API and Pageviews API, with built-in caching, rate limiting, and
robust error handling to ensure reliable data ingestion.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests  # type: ignore
from cachetools import TTLCache  # type: ignore
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class WikiClient:
    """Client for Wikipedia MediaWiki and Pageviews APIs with caching and retry support.

    This client provides a unified interface to Wikipedia's APIs with built-in
    caching, rate limiting, and retry logic to handle API failures gracefully.
    It supports fetching page content, sections, templates, revisions, backlinks,
    and pageview statistics.
    """

    def __init__(
        self,
        base_url: str = "https://en.wikipedia.org/w/api.php",
        pageviews_url: str = "https://wikimedia.org/api/rest_v1/metrics/pageviews",
        cache_ttl: int = 3600,
        max_cache_size: int = 1000,
        rate_limit_delay: float = 0.1,
        max_retries: int = 3,
    ) -> None:
        """Initialize the WikiClient with configuration parameters.

        Args:
            base_url: Base URL for MediaWiki API
            pageviews_url: Base URL for Pageviews API
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum number of cached responses
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url
        self.pageviews_url = pageviews_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

        # Initialize cache with TTL
        self._cache = TTLCache(maxsize=max_cache_size, ttl=cache_ttl)

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "WikiClient/1.0 (https://github.com/yourusername/wikipedia)"}
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

    def get_page_content(
        self,
        title: str,
        format: str = "json",
        action: str = "query",
        prop: str = "extracts",
        exintro: bool = False,
        explaintext: bool = True,
    ) -> Dict[str, Any]:
        """Fetch page content from Wikipedia.

        Args:
            title: Page title to fetch
            format: Response format (default: json)
            action: API action (default: query)
            prop: Properties to fetch (default: extracts)
            exintro: Only extract introduction (default: False)
            explaintext: Return plain text (default: True)

        Returns:
            Dictionary containing page content and metadata
        """
        cache_key = self._get_cache_key("page_content", title=title, exintro=exintro)

        params = {
            "format": format,
            "action": action,
            "prop": prop,
            "titles": title,
            "exintro": exintro,
            "explaintext": explaintext,
        }

        response = self._make_request(self.base_url, params, cache_key)

        # Add timestamp and metadata
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "data": response,
        }

        return result  # type: ignore

    def get_sections(
        self,
        title: str,
        format: str = "json",
        action: str = "parse",
        prop: str = "sections",
    ) -> Dict[str, Any]:
        """Fetch page sections from Wikipedia.

        Args:
            title: Page title to fetch sections for
            format: Response format (default: json)
            action: API action (default: parse)
            prop: Properties to fetch (default: sections)

        Returns:
            Dictionary containing page sections and metadata
        """
        cache_key = self._get_cache_key("sections", title=title)

        params = {
            "format": format,
            "action": action,
            "page": title,
            "prop": prop,
        }

        response = self._make_request(self.base_url, params, cache_key)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "data": response,
        }

        return result  # type: ignore

    def get_templates(
        self,
        title: str,
        format: str = "json",
        action: str = "query",
        prop: str = "templates",
        tlnamespace: int = 10,
        tllimit: int = 500,
    ) -> Dict[str, Any]:
        """Fetch templates used in a Wikipedia page.

        Args:
            title: Page title to fetch templates for
            format: Response format (default: json)
            action: API action (default: query)
            prop: Properties to fetch (default: templates)
            tlnamespace: Template namespace (default: 10)
            tllimit: Maximum number of templates to return (default: 500)

        Returns:
            Dictionary containing page templates and metadata
        """
        cache_key = self._get_cache_key("templates", title=title, tllimit=tllimit)

        params = {
            "format": format,
            "action": action,
            "titles": title,
            "prop": prop,
            "tlnamespace": tlnamespace,
            "tllimit": tllimit,
        }

        response = self._make_request(self.base_url, params, cache_key)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "data": response,
        }

        return result  # type: ignore

    def get_revisions(
        self,
        title: str,
        format: str = "json",
        action: str = "query",
        prop: str = "revisions",
        rvlimit: int = 10,
        rvprop: str = "ids|timestamp|user|comment|size",
    ) -> Dict[str, Any]:
        """Fetch page revision history from Wikipedia.

        Args:
            title: Page title to fetch revisions for
            format: Response format (default: json)
            action: API action (default: query)
            prop: Properties to fetch (default: revisions)
            rvlimit: Number of revisions to fetch (default: 10)
            rvprop: Revision properties to fetch

        Returns:
            Dictionary containing page revisions and metadata
        """
        cache_key = self._get_cache_key("revisions", title=title, rvlimit=rvlimit)

        params = {
            "format": format,
            "action": action,
            "titles": title,
            "prop": prop,
            "rvlimit": rvlimit,
            "rvprop": rvprop,
        }

        response = self._make_request(self.base_url, params, cache_key)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "data": response,
        }

        return result  # type: ignore

    def get_backlinks(
        self,
        title: str,
        format: str = "json",
        action: str = "query",
        list: str = "backlinks",
        blnamespace: int = 0,
        bllimit: int = 50,
    ) -> Dict[str, Any]:
        """Fetch pages that link to the given page.

        Args:
            title: Page title to find backlinks for
            format: Response format (default: json)
            action: API action (default: query)
            list: List type (default: backlinks)
            blnamespace: Namespace to search (default: 0 for main namespace)
            bllimit: Number of backlinks to fetch (default: 50)

        Returns:
            Dictionary containing backlinks and metadata
        """
        cache_key = self._get_cache_key("backlinks", title=title, bllimit=bllimit)

        params = {
            "format": format,
            "action": action,
            "list": list,
            "bltitle": title,
            "blnamespace": blnamespace,
            "bllimit": bllimit,
        }

        response = self._make_request(self.base_url, params, cache_key)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "data": response,
        }

        return result  # type: ignore

    def get_pageviews(
        self,
        title: str,
        start_date: str,
        end_date: str,
        access: str = "all-access",
        agent: str = "user",
        granularity: str = "daily",
    ) -> Dict[str, Any]:
        """Fetch pageview statistics from Wikipedia Pageviews API.

        Args:
            title: Page title to fetch pageviews for
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            access: Access type (all-access, desktop, mobile-app, mobile-web)
            agent: Agent type (user, spider, bot, automated)
            granularity: Data granularity (daily, monthly)

        Returns:
            Dictionary containing pageview statistics and metadata
        """
        cache_key = self._get_cache_key(
            "pageviews",
            title=title,
            start_date=start_date,
            end_date=end_date,
            access=access,
            agent=agent,
        )

        # URL encode the title for the API
        encoded_title = quote(title, safe="")

        url = f"{self.pageviews_url}/per-article/en.wikipedia.org/{access}/{agent}/{encoded_title}/{granularity}/{start_date}/{end_date}"

        response = self._make_request(url, {}, cache_key)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "start_date": start_date,
            "end_date": end_date,
            "access": access,
            "agent": agent,
            "granularity": granularity,
            "data": response,
        }

        return result  # type: ignore

    def get_citations(
        self,
        title: str,
        format: str = "json",
        action: str = "query",
        prop: str = "extlinks",
        ellimit: int = 50,
    ) -> Dict[str, Any]:
        """Fetch external links (citations) from a Wikipedia page.

        Args:
            title: Page title to fetch citations for
            format: Response format (default: json)
            action: API action (default: query)
            prop: Properties to fetch (default: extlinks)
            ellimit: Number of external links to fetch (default: 50)

        Returns:
            Dictionary containing external links and metadata
        """
        cache_key = self._get_cache_key("citations", title=title, ellimit=ellimit)

        params = {
            "format": format,
            "action": action,
            "titles": title,
            "prop": prop,
            "ellimit": ellimit,
        }

        response = self._make_request(self.base_url, params, cache_key)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "data": response,
        }

        return result  # type: ignore

    def get_links(
        self,
        title: str,
        format: str = "json",
        action: str = "query",
        prop: str = "links",
        pllimit: int = 100,
    ) -> Dict[str, Any]:
        """Fetch internal links from a Wikipedia page.

        Args:
            title: Page title to fetch links for
            format: Response format (default: json)
            action: API action (default: query)
            prop: Properties to fetch (default: links)
            pllimit: Number of links to fetch (default: 100)

        Returns:
            Dictionary containing internal links and metadata
        """
        cache_key = self._get_cache_key("links", title=title, pllimit=pllimit)

        params = {
            "format": format,
            "action": action,
            "titles": title,
            "prop": prop,
            "pllimit": pllimit,
        }

        response = self._make_request(self.base_url, params, cache_key)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "data": response,
        }

        return result  # type: ignore

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
