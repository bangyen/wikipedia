"""Central utility module for extracting data from Wikipedia API responses.

Provides a unified interface for parsing complex Wikipedia JSON data formats
into structured lists and strings.
"""

import re
from typing import Any, Dict, List


def extract_sections(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract sections from article data."""
    data = article_data.get("data", {})

    # Check if we already have a list (new format)
    sections = data.get("sections")
    if isinstance(sections, list):
        return sections

    # Try common paths (old format)
    if "parse" in data and "sections" in data["parse"]:
        sections = data["parse"]["sections"]
        if isinstance(sections, list):
            return sections

    query = data.get("query", {})
    if "sections" in query:
        sections = query["sections"]
        if isinstance(sections, list):
            return sections

    return []


def extract_content_text(article_data: Dict[str, Any]) -> str:
    """Extract and clean content text from article data."""
    data = article_data.get("data", {})
    content = ""

    # Check for extract directly (new format)
    if "extract" in data:
        content = data["extract"]
    elif "text" in data and isinstance(data["text"], dict):
        content = data["text"].get("*", "")
        content = re.sub(r"<[^>]+>", "", content)
    # Old format
    elif "parse" in data and "text" in data["parse"]:
        content = data["parse"]["text"].get("*", "")
        content = re.sub(r"<[^>]+>", "", content)
    else:
        pages = data.get("query", {}).get("pages", {})
        if isinstance(pages, dict):
            for page in pages.values():
                if isinstance(page, dict) and "extract" in page:
                    content = page["extract"]
                    break

    return content


def extract_templates(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract templates from article data."""
    data = article_data.get("data", {})

    # Check for direct list (new format)
    templates = data.get("templates")
    if isinstance(templates, list):
        return templates

    query = data.get("query", {})

    # Check directly in query or inside pages (old format)
    if "templates" in query:
        if isinstance(query["templates"], list):
            return query["templates"]
        if isinstance(query["templates"], dict):
            for temps in query["templates"].values():
                if isinstance(temps, dict) and "templates" in temps:
                    templates = temps["templates"]
                    if isinstance(templates, list):
                        return templates

    pages = query.get("pages", {})
    if isinstance(pages, dict):
        for page in pages.values():
            if isinstance(page, dict) and "templates" in page:
                templates = page["templates"]
                if isinstance(templates, list):
                    return templates

    return []


def extract_citations(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract citations (external links) from article data."""
    data = article_data.get("data", {})

    # Check for direct list (new format)
    extlinks = data.get("extlinks")
    if isinstance(extlinks, list):
        return extlinks

    query = data.get("query", {})

    if "extlinks" in query:
        extlinks = query["extlinks"]
        if isinstance(extlinks, list):
            return extlinks
        if isinstance(query["extlinks"], dict):
            for links in query["extlinks"].values():
                if isinstance(links, dict) and "extlinks" in links:
                    extlinks = links["extlinks"]
                    if isinstance(extlinks, list):
                        return extlinks

    pages = query.get("pages", {})
    if isinstance(pages, dict):
        for page in pages.values():
            if isinstance(page, dict) and "extlinks" in page:
                extlinks = page["extlinks"]
                if isinstance(extlinks, list):
                    return extlinks

    return []


def extract_revisions(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract revision history from article data."""
    data = article_data.get("data", {})

    # Check for direct list (new format)
    revisions = data.get("revisions")
    if isinstance(revisions, list):
        return revisions

    query = data.get("query", {})

    if "revisions" in query:
        revisions = query["revisions"]
        if isinstance(revisions, list):
            return revisions
        if isinstance(query["revisions"], dict):
            for revs in query["revisions"].values():
                if isinstance(revs, dict) and "revisions" in revs:
                    revisions = revs["revisions"]
                    if isinstance(revisions, list):
                        return revisions

    pages = query.get("pages", {})
    if isinstance(pages, dict):
        for page in pages.values():
            if isinstance(page, dict) and "revisions" in page:
                revisions = page["revisions"]
                if isinstance(revisions, list):
                    return revisions

    return []


def extract_backlinks(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract backlinks from article data."""
    data = article_data.get("data", {})

    # Check for direct list (new format)
    backlinks = data.get("backlinks")
    if isinstance(backlinks, list):
        return backlinks

    # Old format
    backlinks = data.get("query", {}).get("backlinks", [])
    if isinstance(backlinks, list):
        return backlinks
    return []


def extract_internal_links(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract internal links from article data."""
    data = article_data.get("data", {})

    # Check for direct list (new format)
    links = data.get("links")
    if isinstance(links, list):
        return links

    query = data.get("query", {})

    if "links" in query:
        links_data = query["links"]
        if isinstance(links_data, list):
            return links_data
        if isinstance(query["links"], dict):
            for links in query["links"].values():
                if isinstance(links, dict) and "links" in links:
                    links_data = links["links"]
                    if isinstance(links_data, list):
                        return links_data

    pages = query.get("pages", {})
    if isinstance(pages, dict):
        for page in pages.values():
            if isinstance(page, dict) and "links" in page:
                links_data = page["links"]
                if isinstance(links_data, list):
                    return links_data

    return []


def extract_categories(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract categories from article data."""
    data = article_data.get("data", {})

    # Check for direct list (new format)
    categories = data.get("categories")
    if isinstance(categories, list):
        return categories

    if "parse" in data and "categories" in data["parse"]:
        categories = data["parse"]["categories"]
        if isinstance(categories, list):
            return categories

    query = data.get("query", {})
    pages = query.get("pages", {})
    if isinstance(pages, dict):
        for page in pages.values():
            if isinstance(page, dict) and "categories" in page:
                categories = page["categories"]
                if isinstance(categories, list):
                    return categories

    return []
