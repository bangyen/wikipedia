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

# Sample data for demo purposes
SAMPLE_ARTICLES = {
    "Albert Einstein": {
        "title": "Albert Einstein",
        "maturity_score": 82.5,
        "pillar_scores": {
            "structure": 85.3,
            "sourcing": 88.7,
            "editorial": 76.2,
            "network": 79.4,
        },
    },
    "Python (programming language)": {
        "title": "Python (programming language)",
        "maturity_score": 78.2,
        "pillar_scores": {
            "structure": 82.1,
            "sourcing": 75.6,
            "editorial": 81.3,
            "network": 73.8,
        },
    },
    "Wikipedia": {
        "title": "Wikipedia",
        "maturity_score": 79.8,
        "pillar_scores": {
            "structure": 78.5,
            "sourcing": 84.2,
            "editorial": 74.6,
            "network": 81.9,
        },
    },
    "Stub": {
        "title": "Stub",
        "maturity_score": 45.3,
        "pillar_scores": {
            "structure": 38.2,
            "sourcing": 42.1,
            "editorial": 51.7,
            "network": 48.9,
        },
    },
    "List of colors": {
        "title": "List of colors",
        "maturity_score": 52.7,
        "pillar_scores": {
            "structure": 61.3,
            "sourcing": 35.8,
            "editorial": 48.2,
            "network": 67.5,
        },
    },
}

# Peer groups for comparison
PEER_GROUPS = {
    "science": [
        "Albert Einstein",
        "Machine Learning",
        "Artificial Intelligence",
        "Computer Science",
        "Physics",
        "Mathematics",
        "Chemistry",
        "Biology",
    ],
    "technology": [
        "Python (programming language)",
        "JavaScript",
        "Java (programming language)",
        "C++",
        "Computer Science",
        "Software Engineering",
        "Web Development",
        "Data Science",
    ],
    "general": [
        "Wikipedia",
        "Internet",
        "World Wide Web",
        "Information",
        "Knowledge",
        "Education",
        "Research",
        "Communication",
    ],
}


def get_peer_group(title: str) -> List[str]:
    """Determine peer group for an article based on title."""
    title_lower = title.lower()

    if any(
        keyword in title_lower
        for keyword in ["einstein", "physics", "science", "mathematics"]
    ):
        return PEER_GROUPS["science"]
    elif any(
        keyword in title_lower
        for keyword in ["python", "programming", "computer", "software", "technology"]
    ):
        return PEER_GROUPS["technology"]
    else:
        return PEER_GROUPS["general"]


def fetch_article_data(title: str) -> Optional[Dict[str, Any]]:
    """Fetch comprehensive article data from Wikipedia API."""
    try:
        # Fetch comprehensive article data
        page_content = wiki_client.get_page_content(title)
        sections = wiki_client.get_sections(title)
        templates = wiki_client.get_templates(title)
        revisions = wiki_client.get_revisions(title, rvlimit=20)
        backlinks = wiki_client.get_backlinks(title, bllimit=50)
        citations = wiki_client.get_citations(title, ellimit=50)

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
