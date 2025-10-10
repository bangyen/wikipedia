#!/usr/bin/env python3
"""Demo script for heuristic baseline model.

This script demonstrates the heuristic baseline model by scoring sample
Wikipedia articles and showing the maturity score breakdown.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
from models.baseline import HeuristicBaselineModel  # noqa: E402
from src.ingest.wiki_client import WikiClient  # noqa: E402


def demo_article_scoring() -> None:
    """Demonstrate article scoring with the heuristic baseline model."""
    print("Heuristic Baseline Model Demo")
    print("=" * 40)

    # Initialize model and client
    model = HeuristicBaselineModel()
    client = WikiClient()

    # Sample articles to demonstrate
    sample_articles = [
        "Albert Einstein",  # Should be high quality
        "Python (programming language)",  # Should be high quality
        "Stub",  # Should be low quality
        "Wikipedia",  # Should be high quality
        "List of colors",  # Should be medium quality
    ]

    print(f"Scoring {len(sample_articles)} sample articles...\n")

    results = []

    for title in sample_articles:
        print(f"Processing: {title}")

        try:
            # Fetch comprehensive article data
            page_content = client.get_page_content(title)
            sections = client.get_sections(title)
            templates = client.get_templates(title)
            revisions = client.get_revisions(title, rvlimit=20)
            backlinks = client.get_backlinks(title, bllimit=50)
            citations = client.get_citations(title, ellimit=50)

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

            # Calculate maturity score
            result = model.calculate_maturity_score(article_data)

            # Display results
            print(f"  Maturity Score: {result['maturity_score']:.1f}/100")
            print("  Pillar Scores:")
            for pillar, score in result["pillar_scores"].items():
                print(f"    {pillar.capitalize()}: {score:.1f}")

            # Show top features
            importance = model.get_feature_importance(article_data)
            top_features = list(importance.items())[:5]
            print("  Top Features:")
            for feature, imp in top_features:
                print(f"    {feature}: {imp:.3f}")

            print()

            results.append(
                {
                    "title": title,
                    "maturity_score": result["maturity_score"],
                    "pillar_scores": result["pillar_scores"],
                    "top_features": top_features,
                }
            )

        except Exception as e:
            print(f"  Error: {e}\n")
            continue

    # Summary
    print("Summary:")
    print("-" * 20)
    for result in results:
        print(f"{result['title']}: {result['maturity_score']:.1f}")

    # Save results
    with open("models/demo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDemo results saved to models/demo_results.json")


def demo_weight_analysis() -> None:
    """Demonstrate weight analysis and configuration."""
    print("\nWeight Analysis Demo")
    print("=" * 20)

    model = HeuristicBaselineModel()

    print("Current Pillar Weights:")
    for pillar, weight in model.pillar_weights.items():
        print(f"  {pillar.capitalize()}: {weight:.2f}")

    print("\nTop Feature Weights:")
    sorted_features = sorted(
        model.feature_weights.items(), key=lambda x: abs(x[1]), reverse=True
    )
    for feature, weight in sorted_features[:10]:
        print(f"  {feature}: {weight:.3f}")

    print(f"\nWeights loaded from: {model.weights_file}")


def main() -> bool:
    """Main demo function."""
    try:
        demo_article_scoring()
        demo_weight_analysis()

        print("\nDemo completed successfully!")
        print("The heuristic baseline model is ready for use.")
        print(
            "Run 'python scripts/validate_model.py' to validate correlation with ORES."
        )

    except Exception as e:
        print(f"Demo failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
