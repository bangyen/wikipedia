"""Central utility module for extracting data from Wikipedia API responses.

Provides a unified interface for parsing complex Wikipedia JSON data formats
into structured lists and strings.
"""

import re
from typing import Any, Dict, List


def extract_sections(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract sections from article data."""
    data = article_data.get("data", {})

    # Try common paths
    if "parse" in data and "sections" in data["parse"]:
        sections = data["parse"]["sections"]
        if isinstance(sections, list):
            return sections

    query = data.get("query", {})
    if "sections" in query:
        sections = query["sections"]
        if isinstance(sections, list):
            return sections

    pages = query.get("pages", {})
    for page in pages.values():
        if "sections" in page:
            sections = page["sections"]
            if isinstance(sections, list):
                return sections

    return []


def extract_content_text(article_data: Dict[str, Any]) -> str:
    """Extract and clean content text from article data."""
    data = article_data.get("data", {})
    content = ""

    if "parse" in data and "text" in data["parse"]:
        content = data["parse"]["text"].get("*", "")
        content = re.sub(r"<[^>]+>", "", content)
    else:
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "extract" in page:
                content = page["extract"]
                break

    return content


def extract_templates(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract templates from article data."""
    query = article_data.get("data", {}).get("query", {})

    # Check directly in query or inside pages
    if "templates" in query:
        if isinstance(query["templates"], list):
            return query["templates"]
        if isinstance(query["templates"], dict):
            # Some versions might have templates mapped by page ID
            for temps in query["templates"].values():
                if "templates" in temps:
                    templates = temps["templates"]
                    if isinstance(templates, list):
                        return templates

    pages = query.get("pages", {})
    for page in pages.values():
        if "templates" in page:
            templates = page["templates"]
            if isinstance(templates, list):
                return templates

    return []


def extract_citations(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract citations (external links) from article data."""
    query = article_data.get("data", {}).get("query", {})

    if "extlinks" in query:
        extlinks = query["extlinks"]
        if isinstance(extlinks, list):
            return extlinks
        for links in query["extlinks"].values():
            if "extlinks" in links:
                extlinks = links["extlinks"]
                if isinstance(extlinks, list):
                    return extlinks

    pages = query.get("pages", {})
    for page in pages.values():
        if "extlinks" in page:
            extlinks = page["extlinks"]
            if isinstance(extlinks, list):
                return extlinks

    return []


def extract_revisions(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract revision history from article data."""
    query = article_data.get("data", {}).get("query", {})

    if "revisions" in query:
        revisions = query["revisions"]
        if isinstance(revisions, list):
            return revisions
        for revs in query["revisions"].values():
            if "revisions" in revs:
                revisions = revs["revisions"]
                if isinstance(revisions, list):
                    return revisions

    pages = query.get("pages", {})
    for page in pages.values():
        if "revisions" in page:
            revisions = page["revisions"]
            if isinstance(revisions, list):
                return revisions

    return []


def extract_backlinks(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract backlinks from article data."""
    backlinks = article_data.get("data", {}).get("query", {}).get("backlinks", [])
    if isinstance(backlinks, list):
        return backlinks
    return []


def extract_internal_links(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract internal links from article data."""
    query = article_data.get("data", {}).get("query", {})

    if "links" in query:
        links_data = query["links"]
        if isinstance(links_data, list):
            return links_data
        for links in query["links"].values():
            if "links" in links:
                links_data = links["links"]
                if isinstance(links_data, list):
                    return links_data

    pages = query.get("pages", {})
    for page in pages.values():
        if "links" in page:
            links_data = page["links"]
            if isinstance(links_data, list):
                return links_data

    return []


def extract_categories(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract categories from article data."""
    data = article_data.get("data", {})

    if "parse" in data and "categories" in data["parse"]:
        categories = data["parse"]["categories"]
        if isinstance(categories, list):
            return categories

    query = data.get("query", {})
    pages = query.get("pages", {})
    for page in pages.values():
        if "categories" in page:
            categories = page["categories"]
            if isinstance(categories, list):
                return categories

    return []
