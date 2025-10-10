#!/usr/bin/env python3
"""Command-line interface for Wikipedia article maturity scoring.

This module provides a CLI tool for scoring Wikipedia articles using the
heuristic baseline model. Prints formatted output with color-coded
maturity levels and top contributing features.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
from colorama import Fore, Style, init

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.baseline import HeuristicBaselineModel  # noqa: E402
from wiki_client import WikiClient  # noqa: E402

# Initialize colorama for cross-platform colored output
init(autoreset=True)


def get_maturity_band(score: float) -> str:
    """Convert numeric score to maturity band.

    Args:
        score: Maturity score (0-100)

    Returns:
        Maturity band string
    """
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    elif score >= 20:
        return "Poor"
    else:
        return "Very Poor"


def get_band_color(band: str) -> str:
    """Get color code for maturity band.

    Args:
        band: Maturity band string

    Returns:
        Colorama color code
    """
    color_map = {
        "Excellent": Fore.GREEN,
        "Good": Fore.CYAN,
        "Fair": Fore.YELLOW,
        "Poor": Fore.MAGENTA,
        "Very Poor": Fore.RED,
    }
    return str(color_map.get(band, Fore.WHITE))


def fetch_article_data(title: str) -> Optional[Dict[str, Any]]:
    """Fetch Wikipedia article data for scoring.

    Args:
        title: Article title

    Returns:
        Article data dictionary or None if not found
    """
    try:
        # Initialize wiki client
        wiki_client = WikiClient()

        # Fetch comprehensive article data
        page_content = wiki_client.get_page_content(title)
        sections = wiki_client.get_sections(title)
        templates = wiki_client.get_templates(title)
        revisions = wiki_client.get_revisions(title, rvlimit=20)
        backlinks = wiki_client.get_backlinks(title, bllimit=50)
        citations = wiki_client.get_citations(title, ellimit=50)

        # Combine data into expected format
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
        print(f"Error fetching data for {title}: {e}", file=sys.stderr)
        return None


def print_score_summary(title: str, result: Dict[str, Any]) -> None:
    """Print formatted score summary.

    Args:
        title: Article title
        result: Scoring result dictionary
    """
    # Extract results
    maturity_score = result["maturity_score"]
    pillar_scores = result["pillar_scores"]
    band = get_maturity_band(maturity_score)
    band_color = get_band_color(band)

    # Print header
    print(f"\n{Fore.BLUE}{'='*60}")
    print(f"{Fore.BLUE}Wikipedia Article Maturity Score")
    print(f"{Fore.BLUE}{'='*60}")

    # Print title and main score
    print(f"\n{Fore.WHITE}Article: {Fore.CYAN}{title}")
    print(
        f"{Fore.WHITE}Score: {band_color}{maturity_score:.1f}/100 {Style.BRIGHT}({band}){Style.RESET_ALL}"
    )

    # Print pillar breakdown
    print(f"\n{Fore.WHITE}Pillar Scores:")
    for pillar, score in pillar_scores.items():
        pillar_color = (
            Fore.GREEN if score >= 60 else Fore.YELLOW if score >= 40 else Fore.RED
        )
        print(f"  {Fore.WHITE}{pillar.capitalize():12}: {pillar_color}{score:.1f}/100")

    # Print top contributing features
    print(f"\n{Fore.WHITE}Top Contributing Features:")
    feature_importance = result.get("feature_importance", {})
    if feature_importance:
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            print(
                f"  {Fore.WHITE}{i}. {Fore.CYAN}{feature}: {Fore.YELLOW}{importance:.3f}"
            )
    else:
        print(f"  {Fore.YELLOW}Feature importance not available")

    print(f"\n{Fore.BLUE}{'='*60}")


@click.command()
@click.argument("title", type=str)
@click.option(
    "--json",
    is_flag=True,
    help="Output results in JSON format instead of formatted text",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed scoring information")
def main(title: str, json: bool, verbose: bool) -> None:
    """Score a Wikipedia article for maturity.

    TITLE: Wikipedia article title to score

    Examples:
        wiki-score "Albert Einstein"
        wiki-score "Python (programming language)" --json
        wiki-score "Machine learning" --verbose
    """
    try:
        # Initialize model
        model = HeuristicBaselineModel()

        # Fetch article data
        click.echo(f"Fetching data for '{title}'...", err=True)
        article_data = fetch_article_data(title)

        if not article_data:
            click.echo(
                f"Error: Article '{title}' not found or could not be fetched", err=True
            )
            sys.exit(1)

        # Calculate maturity score
        click.echo("Calculating maturity score...", err=True)
        result = model.calculate_maturity_score(article_data)

        # Get feature importance
        feature_importance = model.get_feature_importance(article_data)
        result["feature_importance"] = feature_importance

        # Output results
        if json:
            # JSON output
            import json as json_module

            output = {
                "title": title,
                "score": result["maturity_score"],
                "band": get_maturity_band(result["maturity_score"]),
                "pillar_scores": result["pillar_scores"],
                "top_features": dict(list(feature_importance.items())[:5]),
            }
            if verbose:
                output.update(
                    {
                        "raw_features": result["raw_features"],
                        "normalized_features": result["normalized_features"],
                        "all_feature_importance": feature_importance,
                    }
                )
            print(json_module.dumps(output, indent=2))
        else:
            # Formatted output
            print_score_summary(title, result)

            if verbose:
                print(f"\n{Fore.WHITE}Detailed Information:")
                print(f"{Fore.WHITE}Raw Features: {len(result['raw_features'])}")
                print(
                    f"{Fore.WHITE}Normalized Features: {len(result['normalized_features'])}"
                )

                # Show all feature importance if verbose
                print(f"\n{Fore.WHITE}All Feature Importance:")
                sorted_features = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )
                for feature, importance in sorted_features:
                    print(f"  {Fore.CYAN}{feature}: {Fore.YELLOW}{importance:.3f}")

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
