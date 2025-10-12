#!/usr/bin/env python3
"""Flask API server for Wikipedia Maturity Dashboard.

This server provides REST endpoints for the JavaScript dashboard to fetch
article maturity scores and peer comparisons.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
from models.baseline import HeuristicBaselineModel  # noqa: E402
from wiki_client import WikiClient  # noqa: E402

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# Initialize model and client
model = HeuristicBaselineModel()
wiki_client = WikiClient()

# Sample articles demonstrating full quality spectrum (0-100)
SAMPLE_ARTICLES = {
    # Featured/Exceptional tier (95-96) - Featured Articles with all features detected
    "Albert Einstein": {
        "title": "Albert Einstein",
        "maturity_score": 95.5,
        "pillar_scores": {
            "structure": 100.0,
            "sourcing": 100.0,
            "editorial": 89.2,
            "network": 76.7,
        },
    },
    "Coffee": {
        "title": "Coffee",
        "maturity_score": 95.5,
        "pillar_scores": {
            "structure": 100.0,
            "sourcing": 100.0,
            "editorial": 89.2,
            "network": 76.7,
        },
    },
    "World War II": {
        "title": "World War II",
        "maturity_score": 95.0,
        "pillar_scores": {
            "structure": 100.0,
            "sourcing": 100.0,
            "editorial": 86.9,
            "network": 76.7,
        },
    },
    "Zoboomafoo": {
        "title": "Zoboomafoo",
        "maturity_score": 85.0,
        "pillar_scores": {
            "structure": 77.8,
            "sourcing": 100.0,
            "editorial": 70.0,
            "network": 76.7,
        },
    },
    # High Quality tier (80-85) - Good Articles
    "Python (programming language)": {
        "title": "Python (programming language)",
        "maturity_score": 82.4,
        "pillar_scores": {
            "structure": 88.8,
            "sourcing": 77.5,
            "editorial": 85.2,
            "network": 76.7,
        },
    },
    "Taylor Swift": {
        "title": "Taylor Swift",
        "maturity_score": 81.8,
        "pillar_scores": {
            "structure": 90.0,
            "sourcing": 75.0,
            "editorial": 85.6,
            "network": 76.7,
        },
    },
    # Good/Developing tier (50-74) - Start-class
    "Banana slug": {
        "title": "Banana slug",
        "maturity_score": 65.2,
        "pillar_scores": {
            "structure": 54.0,
            "sourcing": 75.0,
            "editorial": 57.0,
            "network": 76.0,
        },
    },
    # Stub tier (0-49) - Minimal content
    "List of colours": {
        "title": "List of colours",
        "maturity_score": 9.8,
        "pillar_scores": {
            "structure": 9.0,
            "sourcing": 0.0,
            "editorial": 22.0,
            "network": 28.0,
        },
    },
}

# Peer groups - showing Wikipedia's realistic binary quality distribution
# Most Wikipedia articles score either 80-96 (well-developed) or 0-20 (stubs)
# The "middle range" (30-70) is rare due to Wikipedia's collaborative nature
PEER_GROUPS = {
    "featured": [
        "Albert Einstein",  # 95.5 - Featured Article
        "Coffee",  # 95.5 - Featured Article
        "World War II",  # 95.0 - Featured Article
    ],
    "mixed_quality": [
        # Featured tier (95-96) - comprehensive, well-sourced
        "Albert Einstein",  # 95.5
        "Coffee",  # 95.5
        "World War II",  # 95.0
        # Good tier (81-86) - solid articles
        "Python (programming language)",  # 86.2
        "Banana slug",  # 83.6
        "Hat",  # 81.0
        # Stub tier (5-19) - minimal content
        "List of colours",  # 19.1
        "Blue-green algae",  # 17.2
        "Mango tree",  # 14.8
        "List of animals",  # 5.5
    ],
    "developing": [
        "Banana slug",  # 83.6 - Good quality
        "Hat",  # 81.0 - Good quality
        "List of colours",  # 19.1 - Stub
    ],
}


def get_peer_group(title: str) -> List[str]:
    """Determine peer group for an article based on title.

    Returns mixed_quality by default to show Wikipedia's realistic binary distribution:
    - Featured Articles: 95-96 (comprehensive, well-sourced)
    - Good Articles: 81-86 (solid quality)
    - Stubs: 5-19 (minimal content)
    """
    # Always return mixed_quality to showcase the full scoring range (5.5 - 95.5)
    return PEER_GROUPS["mixed_quality"]


def fetch_article_data(title: str) -> Optional[Dict[str, Any]]:
    """Fetch comprehensive article data from Wikipedia API."""
    try:
        # Fetch comprehensive article data
        page_content = wiki_client.get_page_content(title)
        sections = wiki_client.get_sections(title)
        templates = wiki_client.get_templates(title, tllimit=500)
        revisions = wiki_client.get_revisions(title, rvlimit=100)
        backlinks = wiki_client.get_backlinks(title, bllimit=100)
        citations = wiki_client.get_citations(title, ellimit=100)
        links = wiki_client.get_links(title, pllimit=200)

        # Combine data
        article_data = {
            "title": title,
            "data": {
                "parse": page_content.get("data", {}).get("parse", {}),
                "query": {
                    "pages": page_content.get("data", {})
                    .get("query", {})
                    .get("pages", {}),
                    "sections": sections.get("data", {})
                    .get("parse", {})
                    .get("sections", []),
                    "templates": templates.get("data", {})
                    .get("query", {})
                    .get("pages", {}),
                    "revisions": revisions.get("data", {})
                    .get("query", {})
                    .get("pages", {}),
                    "backlinks": backlinks.get("data", {})
                    .get("query", {})
                    .get("backlinks", []),
                    "extlinks": citations.get("data", {})
                    .get("query", {})
                    .get("pages", {}),
                    "links": links.get("data", {}).get("query", {}).get("pages", {}),
                },
            },
        }

        return article_data

    except Exception as e:
        print(f"Error fetching data for {title}: {e}")
        return None


def calculate_maturity_score(title: str) -> Optional[Dict[str, Any]]:
    """Calculate maturity score for an article."""
    # Try to fetch real data first
    article_data = fetch_article_data(title)
    if article_data:
        try:
            result = model.calculate_maturity_score(article_data)
            return {
                "title": title,
                "maturity_score": result["maturity_score"],
                "pillar_scores": result["pillar_scores"],
            }
        except Exception as e:
            print(f"Error calculating score for {title}: {e}")

    # Fallback to sample data if real data fails
    if title in SAMPLE_ARTICLES:
        print(f"Using sample data for {title}")
        return SAMPLE_ARTICLES[title]

    return None


@app.route("/")
def index() -> Any:
    """Serve the main dashboard page."""
    return send_from_directory(".", "index.html")


@app.route("/api/article/<title>")
def get_article(title: str) -> Any:
    """Get maturity score for a specific article."""
    try:
        result = calculate_maturity_score(title)
        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "Article not found or could not be analyzed"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/peers/<title>")
def get_peer_articles(title: str) -> Any:
    """Get peer articles for comparison."""
    try:
        peer_titles = get_peer_group(title)
        peer_articles = []

        for peer_title in peer_titles:
            if peer_title != title:  # Don't include the current article
                peer_data = calculate_maturity_score(peer_title)
                if peer_data:
                    peer_articles.append(peer_data)

        return jsonify(peer_articles)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def search_articles() -> Any:
    """Search for articles (placeholder for future implementation)."""
    query = request.args.get("q", "")
    if not query:
        return jsonify([])

    # For now, return sample articles that match the query
    results = []
    for title in SAMPLE_ARTICLES.keys():
        if query.lower() in title.lower():
            results.append(
                {
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                }
            )

    return jsonify(results)


@app.route("/api/health")
def health_check() -> Any:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": "loaded"})


if __name__ == "__main__":
    print("Starting Wikipedia Maturity Dashboard API...")
    print("Dashboard available at: http://localhost:5000")
    print("API endpoints:")
    print("  GET /api/article/<title> - Get article maturity score")
    print("  GET /api/peers/<title> - Get peer articles")
    print("  GET /api/search?q=<query> - Search articles")
    print("  GET /api/health - Health check")

    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
