"""Feature extraction functions for Wikipedia articles.

This module extracts normalized numeric features from raw Wikipedia article JSON
data, including structural analysis, citation patterns, editorial activity,
and network connectivity metrics.
"""

import json
import math
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np


def structure_features(article_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract structural features from Wikipedia article data.
    
    Analyzes article structure including sections, depth hierarchy, templates,
    and content organization patterns.
    
    Args:
        article_data: Raw Wikipedia article JSON data containing sections,
                     content, and metadata
            
    Returns:
        Dictionary of normalized structural features including:
        - section_count: Total number of sections
        - avg_section_depth: Average section nesting depth
        - max_section_depth: Maximum section nesting depth
        - template_count: Number of templates used
        - content_length: Total content length in characters
        - sections_per_1k_chars: Sections per 1000 characters
        - depth_variance: Variance in section depths
        - has_infobox: Whether article has infobox template
        - has_categories: Whether article has category templates
        - has_navbox: Whether article has navigation templates
    """
    features = {}
    
    # Extract sections data
    sections = _extract_sections(article_data)
    content_text = _extract_content_text(article_data)
    templates = _extract_templates(article_data)
    
    # Basic counts
    features["section_count"] = float(len(sections))
    features["template_count"] = float(len(templates))
    features["content_length"] = float(len(content_text))
    
    # Section depth analysis
    if sections:
        depths = [section.get("level", 1) for section in sections]
        features["avg_section_depth"] = float(np.mean(depths))
        features["max_section_depth"] = float(max(depths))
        features["depth_variance"] = float(np.var(depths))
        features["min_section_depth"] = float(min(depths))
    else:
        features["avg_section_depth"] = 0.0
        features["max_section_depth"] = 0.0
        features["depth_variance"] = 0.0
        features["min_section_depth"] = 0.0
    
    # Normalized metrics
    if features["content_length"] > 0:
        features["sections_per_1k_chars"] = (features["section_count"] * 1000) / features["content_length"]
    else:
        features["sections_per_1k_chars"] = 0.0
    
    # Template analysis
    template_names = [t.get("title", "").lower() for t in templates]
    features["has_infobox"] = float(any("infobox" in name for name in template_names))
    features["has_categories"] = float(any("category" in name for name in template_names))
    features["has_navbox"] = float(any("navbox" in name for name in template_names))
    features["has_stub"] = float(any("stub" in name for name in template_names))
    features["has_disambiguation"] = float(any("disambiguation" in name for name in template_names))
    features["has_taxonbar"] = float(any("taxonbar" in name for name in template_names))
    
    # Content structure analysis
    features["has_references"] = float("references" in content_text.lower())
    features["has_external_links"] = float("external links" in content_text.lower())
    features["has_see_also"] = float("see also" in content_text.lower())
    
    # Log scaling for large values
    features["log_content_length"] = math.log(max(features["content_length"], 1))
    features["log_section_count"] = math.log(max(features["section_count"], 1))
    
    return features


def sourcing_features(article_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract citation and sourcing features from Wikipedia article data.
    
    Analyzes citation patterns, external links, reference density, and
    source quality indicators.
    
    Args:
        article_data: Raw Wikipedia article JSON data containing citations,
                     external links, and content
            
    Returns:
        Dictionary of normalized sourcing features including:
        - citation_count: Total number of citations
        - external_link_count: Number of external links
        - citations_per_1k_tokens: Citations per 1000 tokens
        - external_links_per_1k_tokens: External links per 1000 tokens
        - citation_density: Citation density score
        - has_reliable_sources: Presence of reliable source indicators
        - academic_source_ratio: Ratio of academic sources
        - news_source_ratio: Ratio of news sources
        - gov_source_ratio: Ratio of government sources
        - org_source_ratio: Ratio of organization sources
    """
    features = {}
    
    # Extract data
    citations = _extract_citations(article_data)
    external_links = _extract_external_links(article_data)
    content_text = _extract_content_text(article_data)
    
    # Basic counts
    features["citation_count"] = float(len(citations))
    features["external_link_count"] = float(len(external_links))
    
    # Token estimation (rough approximation)
    token_count = len(content_text.split())
    features["token_count"] = float(token_count)
    
    # Normalized metrics
    if token_count > 0:
        features["citations_per_1k_tokens"] = (features["citation_count"] * 1000) / token_count
        features["external_links_per_1k_tokens"] = (features["external_link_count"] * 1000) / token_count
    else:
        features["citations_per_1k_tokens"] = 0.0
        features["external_links_per_1k_tokens"] = 0.0
    
    # Citation density (citations per character)
    if len(content_text) > 0:
        features["citation_density"] = features["citation_count"] / len(content_text)
    else:
        features["citation_density"] = 0.0
    
    # Source type analysis
    academic_count = 0
    news_count = 0
    gov_count = 0
    org_count = 0
    
    for link in external_links:
        url = link.get("url", "").lower()
        if any(domain in url for domain in [".edu", ".ac.", "scholar", "researchgate"]):
            academic_count += 1
        elif any(domain in url for domain in [".com/news", "bbc", "cnn", "reuters", "ap.org"]):
            news_count += 1
        elif any(domain in url for domain in [".gov", ".mil"]):
            gov_count += 1
        elif any(domain in url for domain in [".org", ".net"]) and not any(domain in url for domain in [".edu", ".gov", ".mil"]):
            org_count += 1
    
    total_sources = len(external_links)
    if total_sources > 0:
        features["academic_source_ratio"] = academic_count / total_sources
        features["news_source_ratio"] = news_count / total_sources
        features["gov_source_ratio"] = gov_count / total_sources
        features["org_source_ratio"] = org_count / total_sources
    else:
        features["academic_source_ratio"] = 0.0
        features["news_source_ratio"] = 0.0
        features["gov_source_ratio"] = 0.0
        features["org_source_ratio"] = 0.0
    
    # Source quality indicators
    features["has_reliable_sources"] = float(
        features["academic_source_ratio"] > 0.1 or 
        features["gov_source_ratio"] > 0.1 or
        features["news_source_ratio"] > 0.2
    )
    
    # Log scaling
    features["log_citation_count"] = math.log(max(features["citation_count"], 1))
    features["log_external_link_count"] = math.log(max(features["external_link_count"], 1))
    
    return features


def editorial_features(article_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract editorial activity features from Wikipedia article data.
    
    Analyzes editor participation, revision patterns, and collaborative
    editing indicators over time.
    
    Args:
        article_data: Raw Wikipedia article JSON data containing revisions,
                     editor information, and timestamps
            
    Returns:
        Dictionary of normalized editorial features including:
        - total_editors: Total number of unique editors
        - editors_90_days: Editors active in last 90 days
        - editors_30_days: Editors active in last 30 days
        - editors_7_days: Editors active in last 7 days
        - total_revisions: Total number of revisions
        - revisions_per_editor: Average revisions per editor
        - editor_diversity: Diversity of editor participation
        - recent_activity_score: Recent editing activity score
        - bot_edit_ratio: Ratio of bot edits
        - anonymous_edit_ratio: Ratio of anonymous edits
        - major_editor_ratio: Ratio of edits by major contributors
    """
    features = {}
    
    # Extract revisions data
    revisions = _extract_revisions(article_data)
    
    if not revisions:
        # Return zero values if no revisions
        return {
            "total_editors": 0.0,
            "editors_90_days": 0.0,
            "editors_30_days": 0.0,
            "editors_7_days": 0.0,
            "total_revisions": 0.0,
            "revisions_per_editor": 0.0,
            "editor_diversity": 0.0,
            "recent_activity_score": 0.0,
            "bot_edit_ratio": 0.0,
            "anonymous_edit_ratio": 0.0,
            "major_editor_ratio": 0.0,
            "log_total_editors": 0.0,
            "log_total_revisions": 0.0,
        }
    
    # Time-based analysis
    now = datetime.now(timezone.utc)
    cutoff_90_days = now - timedelta(days=90)
    cutoff_30_days = now - timedelta(days=30)
    cutoff_7_days = now - timedelta(days=7)
    
    # Editor analysis
    editors = set()
    editors_90_days = set()
    editors_30_days = set()
    editors_7_days = set()
    
    bot_edits = 0
    anonymous_edits = 0
    editor_contributions = {}
    
    for revision in revisions:
        editor = revision.get("user", "")
        timestamp_str = revision.get("timestamp", "")
        
        if editor:
            editors.add(editor)
            editor_contributions[editor] = editor_contributions.get(editor, 0) + 1
            
            # Check if editor is bot
            if editor.lower().endswith("bot") or "bot" in editor.lower():
                bot_edits += 1
            elif editor.startswith("IP:"):
                anonymous_edits += 1
        
        # Parse timestamp
        try:
            if timestamp_str:
                # Handle different timestamp formats
                if "T" in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                
                if timestamp >= cutoff_90_days:
                    editors_90_days.add(editor)
                if timestamp >= cutoff_30_days:
                    editors_30_days.add(editor)
                if timestamp >= cutoff_7_days:
                    editors_7_days.add(editor)
        except (ValueError, TypeError):
            # Skip invalid timestamps
            continue
    
    # Basic counts
    features["total_editors"] = float(len(editors))
    features["editors_90_days"] = float(len(editors_90_days))
    features["editors_30_days"] = float(len(editors_30_days))
    features["editors_7_days"] = float(len(editors_7_days))
    features["total_revisions"] = float(len(revisions))
    
    # Ratios
    if features["total_revisions"] > 0:
        features["bot_edit_ratio"] = bot_edits / features["total_revisions"]
        features["anonymous_edit_ratio"] = anonymous_edits / features["total_revisions"]
    else:
        features["bot_edit_ratio"] = 0.0
        features["anonymous_edit_ratio"] = 0.0
    
    # Editor diversity (Gini coefficient approximation)
    if editor_contributions:
        contributions = list(editor_contributions.values())
        contributions.sort()
        n = len(contributions)
        cumsum = np.cumsum(contributions)
        features["editor_diversity"] = float(1 - (2 * np.sum(cumsum)) / (n * cumsum[-1]) + 1/n)
    else:
        features["editor_diversity"] = 0.0
    
    # Major contributors (top 10% of editors by contribution)
    if editor_contributions:
        sorted_contributions = sorted(editor_contributions.values(), reverse=True)
        major_threshold = sorted_contributions[int(len(sorted_contributions) * 0.1)]
        major_edits = sum(contrib for contrib in editor_contributions.values() if contrib >= major_threshold)
        features["major_editor_ratio"] = major_edits / features["total_revisions"]
    else:
        features["major_editor_ratio"] = 0.0
    
    # Recent activity score (weighted by recency)
    recent_score = 0.0
    for revision in revisions[-10:]:  # Last 10 revisions
        timestamp_str = revision.get("timestamp", "")
        try:
            if timestamp_str:
                if "T" in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                
                days_ago = (now - timestamp).days
                if days_ago <= 7:
                    recent_score += 1.0
                elif days_ago <= 30:
                    recent_score += 0.5
                elif days_ago <= 90:
                    recent_score += 0.2
        except (ValueError, TypeError):
            continue
    
    features["recent_activity_score"] = recent_score
    
    # Average revisions per editor
    if features["total_editors"] > 0:
        features["revisions_per_editor"] = features["total_revisions"] / features["total_editors"]
    else:
        features["revisions_per_editor"] = 0.0
    
    # Log scaling
    features["log_total_editors"] = math.log(max(features["total_editors"], 1))
    features["log_total_revisions"] = math.log(max(features["total_revisions"], 1))
    
    return features


def network_features(article_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract network connectivity features from Wikipedia article data.
    
    Analyzes inbound and outbound links, page connectivity, and network
    centrality indicators.
    
    Args:
        article_data: Raw Wikipedia article JSON data containing backlinks,
                     internal links, and connectivity information
            
    Returns:
        Dictionary of normalized network features including:
        - inbound_links: Number of pages linking to this article
        - outbound_links: Number of pages this article links to
        - internal_links: Number of internal Wikipedia links
        - external_links: Number of external links
        - link_density: Link density score
        - connectivity_score: Overall connectivity score
        - hub_score: Hub-like connectivity score
        - authority_score: Authority-like connectivity score
        - link_balance: Balance between inbound and outbound links
        - network_centrality: Network centrality approximation
    """
    features = {}
    
    # Extract link data
    backlinks = _extract_backlinks(article_data)
    internal_links = _extract_internal_links(article_data)
    external_links = _extract_external_links(article_data)
    content_text = _extract_content_text(article_data)
    
    # Basic counts
    features["inbound_links"] = float(len(backlinks))
    features["outbound_links"] = float(len(internal_links))
    features["internal_links"] = float(len(internal_links))
    features["external_links"] = float(len(external_links))
    
    # Total links
    total_links = features["inbound_links"] + features["outbound_links"]
    features["total_links"] = total_links
    
    # Link density (links per character)
    if len(content_text) > 0:
        features["link_density"] = total_links / len(content_text)
    else:
        features["link_density"] = 0.0
    
    # Connectivity score (normalized combination of inbound and outbound)
    if total_links > 0:
        features["connectivity_score"] = (
            (features["inbound_links"] * 0.6 + features["outbound_links"] * 0.4) / total_links
        )
    else:
        features["connectivity_score"] = 0.0
    
    # Hub and authority scores
    if features["outbound_links"] > 0:
        features["hub_score"] = features["outbound_links"] / (features["outbound_links"] + features["inbound_links"])
    else:
        features["hub_score"] = 0.0
    
    if features["inbound_links"] > 0:
        features["authority_score"] = features["inbound_links"] / (features["outbound_links"] + features["inbound_links"])
    else:
        features["authority_score"] = 0.0
    
    # Link balance (how balanced inbound vs outbound)
    if total_links > 0:
        features["link_balance"] = 1.0 - abs(features["inbound_links"] - features["outbound_links"]) / total_links
    else:
        features["link_balance"] = 0.0
    
    # Network centrality approximation (based on link counts)
    if total_links > 0:
        # Simple centrality based on total connections
        features["network_centrality"] = math.log(total_links + 1) / 10.0  # Normalized
    else:
        features["network_centrality"] = 0.0
    
    # Link type ratios
    if total_links > 0:
        features["internal_link_ratio"] = features["internal_links"] / total_links
        features["external_link_ratio"] = features["external_links"] / total_links
    else:
        features["internal_link_ratio"] = 0.0
        features["external_link_ratio"] = 0.0
    
    # Log scaling for large values
    features["log_inbound_links"] = math.log(max(features["inbound_links"], 1))
    features["log_outbound_links"] = math.log(max(features["outbound_links"], 1))
    features["log_total_links"] = math.log(max(total_links, 1))
    
    return features


# Helper functions for data extraction

def _extract_sections(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract sections from article data."""
    sections = []
    
    # Try different possible paths for sections data
    data = article_data.get("data", {})
    
    if "parse" in data and "sections" in data["parse"]:
        sections = data["parse"]["sections"]
    elif "query" in data and "pages" in data["query"]:
        for page_id, page_data in data["query"]["pages"].items():
            if "sections" in page_data:
                sections = page_data["sections"]
                break
    
    return sections if isinstance(sections, list) else []


def _extract_content_text(article_data: Dict[str, Any]) -> str:
    """Extract content text from article data."""
    content = ""
    
    data = article_data.get("data", {})
    
    if "parse" in data and "text" in data["parse"]:
        content = data["parse"]["text"].get("*", "")
    elif "query" in data and "pages" in data["query"]:
        for page_id, page_data in data["query"]["pages"].items():
            if "extract" in page_data:
                content = page_data["extract"]
                break
    
    return content if isinstance(content, str) else ""


def _extract_templates(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract templates from article data."""
    templates = []
    
    data = article_data.get("data", {})
    
    if "query" in data and "pages" in data["query"]:
        for page_id, page_data in data["query"]["pages"].items():
            if "templates" in page_data:
                templates = page_data["templates"]
                break
    
    return templates if isinstance(templates, list) else []


def _extract_citations(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract citations from article data."""
    citations = []
    
    data = article_data.get("data", {})
    
    if "query" in data and "pages" in data["query"]:
        for page_id, page_data in data["query"]["pages"].items():
            if "extlinks" in page_data:
                citations = page_data["extlinks"]
                break
    
    return citations if isinstance(citations, list) else []


def _extract_external_links(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract external links from article data."""
    return _extract_citations(article_data)  # Same as citations


def _extract_revisions(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract revisions from article data."""
    revisions = []
    
    data = article_data.get("data", {})
    
    if "query" in data and "pages" in data["query"]:
        for page_id, page_data in data["query"]["pages"].items():
            if "revisions" in page_data:
                revisions = page_data["revisions"]
                break
    
    return revisions if isinstance(revisions, list) else []


def _extract_backlinks(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract backlinks from article data."""
    backlinks = []
    
    data = article_data.get("data", {})
    
    if "query" in data and "backlinks" in data["query"]:
        backlinks = data["query"]["backlinks"]
    
    return backlinks if isinstance(backlinks, list) else []


def _extract_internal_links(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract internal links from article data."""
    internal_links = []
    
    data = article_data.get("data", {})
    
    if "query" in data and "pages" in data["query"]:
        for page_id, page_data in data["query"]["pages"].items():
            if "links" in page_data:
                internal_links = page_data["links"]
                break
    
    return internal_links if isinstance(internal_links, list) else []
