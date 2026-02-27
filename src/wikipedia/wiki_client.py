"""Wikipedia API client with caching, rate limiting, and retry logic.

This module provides a comprehensive client for interacting with Wikipedia's
MediaWiki API and Pageviews API, with built-in caching, rate limiting, and
robust error handling to ensure reliable data ingestion.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
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
        use_disk_cache: bool = True,
        cache_dir: str = ".wiki_cache",
    ) -> None:
        """Initialize the WikiClient with configuration parameters.

        Args:
            base_url: Base URL for MediaWiki API
            pageviews_url: Base URL for Pageviews API
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum number of cached responses
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
            use_disk_cache: Whether to use persistent disk cache
            cache_dir: Directory for disk cache
        """
        self.base_url = base_url
        self.pageviews_url = pageviews_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.use_disk_cache = use_disk_cache
        self.cache_dir = Path(cache_dir)

        if self.use_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache with TTL
        self._cache: TTLCache[str, Dict[str, Any]] = TTLCache(
            maxsize=max_cache_size, ttl=cache_ttl
        )

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
        # Check memory cache first
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore

        # Check disk cache
        if self.use_disk_cache and cache_key:
            disk_result = self._read_from_disk(cache_key)
            if disk_result:
                self._cache[cache_key] = disk_result
                return disk_result

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
            if self.use_disk_cache:
                self._write_to_disk(cache_key, result)

        return result  # type: ignore

    def _read_from_disk(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Read a response from disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return cast(Optional[Dict[str, Any]], json.load(f))
            except Exception:
                return None
        return None

    def _write_to_disk(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Write a response to disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _get_cache_key(self, method: str, **kwargs: Any) -> str:
        """Generate a cache key for the given method and parameters."""
        # Sort kwargs for consistent cache keys
        sorted_kwargs = sorted(kwargs.items())
        key_data = f"{method}:{sorted_kwargs}"
        return str(hash(key_data))

    def get_page_content(
        self,
        title: str,
        exintro: bool = False,
        explaintext: bool = True,
    ) -> Dict[str, Any]:
        """Fetch page content from Wikipedia.

        Args:
            title: Page title to fetch
            exintro: Only extract introduction (default: False)
            explaintext: Return plain text (default: True)

        Returns:
            Dictionary containing page content and metadata
        """
        cache_key = self._get_cache_key("page_content", title=title, exintro=exintro)

        params = {
            "format": "json",
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "exintro": exintro,
            "explaintext": explaintext,
        }

        response = self._make_request(self.base_url, params, cache_key)

        # Extract the page content from the response
        pages = response.get("query", {}).get("pages", {})
        page_content = next(iter(pages.values())) if pages else {}

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            **page_content,
        }

    def get_sections(
        self,
        title: str,
    ) -> List[Dict[str, Any]]:
        """Fetch page sections from Wikipedia.

        Args:
            title: Page title to fetch sections for

        Returns:
            List of dictionaries containing page sections
        """
        parsed = self.parse_page(title, prop="sections")
        sections = parsed.get("sections", [])
        return sections if isinstance(sections, list) else []

    def parse_page(
        self,
        title: str,
        prop: str = "text|sections|templates|categories|images|externallinks",
    ) -> Dict[str, Any]:
        """Fetch full parsed page data from Wikipedia.

        Args:
            title: Page title to parse
            prop: Properties to fetch (text, sections, templates, etc.)

        Returns:
            Dictionary containing parsed page data and metadata
        """
        cache_key = self._get_cache_key("parse", title=title, prop=prop)

        params = {
            "format": "json",
            "action": "parse",
            "page": title,
            "prop": prop,
        }

        response = self._make_request(self.base_url, params, cache_key)
        parsed_data = response.get("parse", {})

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            **parsed_data,
        }

    def get_templates(
        self,
        title: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Fetch templates used in a Wikipedia page.

        Args:
            title: Page title to fetch templates for
            limit: Number of templates to fetch (default: 500)

        Returns:
            List of dictionaries containing templates
        """
        cache_key = self._get_cache_key("templates", title=title, limit=limit)

        params = {
            "format": "json",
            "action": "query",
            "titles": title,
            "prop": "templates",
            "tlnamespace": 10,
            "tllimit": limit,
        }

        response = self._make_request(self.base_url, params, cache_key)

        pages = response.get("query", {}).get("pages", {})
        page = next(iter(pages.values())) if pages else {}
        templates = page.get("templates", [])
        return templates if isinstance(templates, list) else []

    def get_revisions(
        self,
        title: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Fetch page revision history from Wikipedia.

        Args:
            title: Page title to fetch revisions for
            limit: Number of revisions to fetch (default: 10)

        Returns:
            List of dictionaries containing revisions
        """
        cache_key = self._get_cache_key("revisions", title=title, limit=limit)

        params = {
            "format": "json",
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvlimit": limit,
            "rvprop": "ids|timestamp|user|comment|size",
        }

        response = self._make_request(self.base_url, params, cache_key)

        pages = response.get("query", {}).get("pages", {})
        page = next(iter(pages.values())) if pages else {}
        revisions = page.get("revisions", [])
        return revisions if isinstance(revisions, list) else []

    def get_backlinks(
        self,
        title: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch pages that link to the given page.

        Args:
            title: Page title to find backlinks for
            limit: Number of backlinks to fetch (default: 50)

        Returns:
            List of dictionaries containing backlinks
        """
        cache_key = self._get_cache_key("backlinks", title=title, limit=limit)

        params = {
            "format": "json",
            "action": "query",
            "list": "backlinks",
            "bltitle": title,
            "blnamespace": 0,
            "bllimit": limit,
        }

        response = self._make_request(self.base_url, params, cache_key)
        backlinks = response.get("query", {}).get("backlinks", [])
        return backlinks if isinstance(backlinks, list) else []

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

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "start_date": start_date,
            "end_date": end_date,
            "access": access,
            "agent": agent,
            "granularity": granularity,
            "items": response.get("items", []),
        }

    def get_citations(
        self,
        title: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch external links (citations) from a Wikipedia page.

        Args:
            title: Page title to fetch citations for
            limit: Number of external links to fetch (default: 50)

        Returns:
            List of dictionaries containing external links
        """
        cache_key = self._get_cache_key("citations", title=title, limit=limit)

        params = {
            "format": "json",
            "action": "query",
            "titles": title,
            "prop": "extlinks",
            "ellimit": limit,
        }

        response = self._make_request(self.base_url, params, cache_key)

        pages = response.get("query", {}).get("pages", {})
        page = next(iter(pages.values())) if pages else {}
        extlinks = page.get("extlinks", [])
        return extlinks if isinstance(extlinks, list) else []

    def get_links(
        self,
        title: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch internal links from a Wikipedia page.

        Args:
            title: Page title to fetch links for
            limit: Number of links to fetch (default: 100)

        Returns:
            List of dictionaries containing internal links
        """
        cache_key = self._get_cache_key("links", title=title, limit=limit)

        params = {
            "format": "json",
            "action": "query",
            "titles": title,
            "prop": "links",
            "pllimit": limit,
        }

        response = self._make_request(self.base_url, params, cache_key)

        pages = response.get("query", {}).get("pages", {})
        page = next(iter(pages.values())) if pages else {}
        links = page.get("links", [])
        return links if isinstance(links, list) else []

    def get_categories(
        self,
        title: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Fetch categories for a Wikipedia page.

        Args:
            title: Page title to fetch categories for
            limit: Number of categories to fetch (default: 500)

        Returns:
            List of dictionaries containing categories
        """
        cache_key = self._get_cache_key("categories", title=title, limit=limit)

        params = {
            "format": "json",
            "action": "query",
            "titles": title,
            "prop": "categories",
            "cllimit": limit,
        }

        response = self._make_request(self.base_url, params, cache_key)

        pages = response.get("query", {}).get("pages", {})
        page = next(iter(pages.values())) if pages else {}
        categories = page.get("categories", [])
        return categories if isinstance(categories, list) else []

    def get_category_members(
        self,
        title: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Fetch pages belonging to a Wikipedia category.

        Args:
            title: Category title (with or without 'Category:' prefix)
            limit: Number of members to fetch (default: 500)

        Returns:
            List of dictionaries containing category members
        """
        # Ensure category name has the prefix
        if not title.startswith("Category:"):
            title = f"Category:{title}"

        cache_key = self._get_cache_key(
            "category_members",
            title=title,
            limit=limit,
        )

        params = {
            "format": "json",
            "action": "query",
            "list": "categorymembers",
            "cmtitle": title,
            "cmtype": "page",
            "cmlimit": limit,
        }

        response = self._make_request(self.base_url, params, cache_key)
        members = response.get("query", {}).get("categorymembers", [])
        return members if isinstance(members, list) else []

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
