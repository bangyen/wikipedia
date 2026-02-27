#!/usr/bin/env python3
"""
Audit script for Wikipedia article maturity scoring.

Downloads data for a set of well-known articles (if not already local)
and runs the HeuristicBaselineModel to calculate maturity scores.
"""

import json
import logging

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wikipedia.models.baseline import HeuristicBaselineModel
from wikipedia.wiki_client import WikiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Articles to audit
# Selected to represent a range of quality and topics
ARTICLES = [
    "Python (programming language)",  # High quality, tech
    "United States",  # High quality, geography, long
    "Douglas Adams",  # Good quality, biography
    "Glitch art",  # Medium quality, art (renamed from Glitch due to disambig)
    "Lorem ipsum",  # Short/Stub-like
]

AUDIT_DIR = Path("audit_data")


def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Recursively update dictionary."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def fetch_full_article_data(client: WikiClient, title: str) -> Dict[str, Any]:
    """Fetch all necessary data for an article from Wikipedia APIs."""
    # Silent fetching for audit

    # Initialize container for merged data
    # We want to mimic the structure: {'data': {'query': ..., 'parse': ...}}
    # as `extract_features` usually looks into `article_data['data']`
    full_data: Dict[str, Any] = {"query": {}, "parse": {}}

    # Helper to merge response data
    def merge(data: Any, key: Optional[str] = None) -> None:
        if isinstance(data, list):
            full_data["query"][key] = data
        elif isinstance(data, dict):
            deep_update(full_data, data)

    try:
        # 1. Full Page Data (actions=parse) - gets text, sections, categories
        full_data["parse"] = client.parse_page(title)

        # 2. Page Content metadata - action=query
        merge(client.get_page_content(title))

        # 3. Templates - action=query
        merge(client.get_templates(title), "templates")

        # 4. Revisions - action=query
        merge(client.get_revisions(title, limit=500), "revisions")

        # 5. Citations (Extlinks) - action=query
        merge(client.get_citations(title, limit=500), "extlinks")

        # 6. Internal Links - action=query
        merge(client.get_links(title, limit=500), "links")

        # 7. Backlinks - action=query
        merge(client.get_backlinks(title, limit=500), "backlinks")

        # 8. Pageviews (optional, usually separate)
        # Not strictly needed for current baseline features, skipping to save time/complexity

    except Exception as e:
        logger.error(f"Error fetching data for {title}: {e}")
        return {}

    return {"title": title, "data": full_data}


def main() -> None:
    """Main execution function."""

    # Ensure audit directory exists
    AUDIT_DIR.mkdir(exist_ok=True)

    # Initialize client and model
    client = WikiClient(rate_limit_delay=0.1)  # Be nice to API
    model = HeuristicBaselineModel()

    print(f"{'Article':<35} | {'Score':<10} | {'Quality Estimate':<15}")
    print("-" * 66)

    for title in ARTICLES:
        file_path = AUDIT_DIR / f"{title.replace(' ', '_').replace('/', '-')}.json"

        article_data = {}

        # Load local or fetch remote
        if file_path.exists():
            # Load local if available
            try:
                with open(file_path, "r") as f:
                    article_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupt file {file_path}, re-fetching.")

        if not article_data:
            article_data = fetch_full_article_data(client, title)
            if article_data:
                with open(file_path, "w") as f:
                    json.dump(article_data, f, indent=2)
            else:
                print(f"{title:<35} | {'ERROR':<10} | {'-'}")
                continue

        # Calculate score
        try:
            result = model.calculate_maturity_score(article_data)
            score = result["maturity_score"]

            # Simple quality estimation mapping
            # Adjusted quality estimation mapping
            if score >= 70:
                quality = "Featured/GA"
            elif score >= 50:
                quality = "B-Class"
            elif score >= 30:
                quality = "C-Class"
            elif score >= 15:
                quality = "Start"
            else:
                quality = "Stub"

            print(f"{title:<35} | {score:<10.2f} | {quality:<15}")

            # Optional: Print pillar breakdown for debugging
            pillar_scores = result["pillar_scores"]
            print(f"  > Pillars: {pillar_scores}")

        except Exception as e:
            logger.error(f"Error scoring {title}: {e}")
            print(f"{title:<35} | {'ERROR':<10} | {'-'}")


if __name__ == "__main__":
    main()
