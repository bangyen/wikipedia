#!/usr/bin/env python3
"""Correlation analysis script for Wikipedia article features.

This script analyzes feature correlations in your dataset to identify:
1. Redundant feature pairs (high correlation)
2. Weak features (low correlation to others)
3. Overall multicollinearity
4. Features recommended for removal

Run with: python scripts/analyze_correlations.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wikipedia.features.correlation_analysis import CorrelationAnalyzer  # noqa: E402
from wikipedia.features.extractors import (  # noqa: E402
    editorial_features,
    network_features,
    sourcing_features,
    structure_features,
)
from wikipedia.wiki_client import WikiClient  # noqa: E402
from typing import Optional  # noqa: E402


def fetch_article_data(client: WikiClient, title: str) -> Optional[Dict[str, Any]]:
    """Fetch article data helper."""
    try:
        page_content = client.get_page_content(title)
        sections = client.get_sections(title)
        templates = client.get_templates(title)
        revisions = client.get_revisions(title, limit=20)
        backlinks = client.get_backlinks(title, limit=50)
        citations = client.get_citations(title, limit=50)

        return {
            "title": title,
            "data": {
                "parse": page_content,
                "query": {
                    "pages": {page_content.get("pageid", "0"): page_content},
                    "sections": sections,
                    "templates": templates,
                    "revisions": revisions,
                    "backlinks": backlinks,
                    "extlinks": citations,
                },
            },
        }
    except Exception:
        return None


def extract_all_features(article_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract all features from article data."""
    features = {}
    features.update(structure_features(article_data))
    features.update(sourcing_features(article_data))
    features.update(editorial_features(article_data))
    features.update(network_features(article_data))
    return features


def fetch_sample_articles(count: int = 50) -> List[Dict[str, Any]]:
    """Fetch sample articles for correlation analysis.

    Args:
        count: Number of articles to fetch.

    Returns:
        List of article data dictionaries.
    """
    client = WikiClient()

    # Sample a diverse set of articles
    sample_titles = [
        "Albert Einstein",
        "Python (programming language)",
        "World War II",
        "Quantum mechanics",
        "Climate change",
        "Artificial intelligence",
        "Shakespeare",
        "United Nations",
        "Newton's laws of motion",
        "Photosynthesis",
        "European Union",
        "Atomic theory",
        "Computer science",
        "Biology",
        "Mozart",
        "Democracy",
        "History of science",
        "Relativity",
        "Gravity",
        "DNA",
        "Economics",
        "Philosophy",
        "Engineering",
        "Medicine",
        "Physics",
    ]

    articles = []
    for title in sample_titles[:count]:
        try:
            article_data = fetch_article_data(client, title)
            if article_data:
                articles.append(article_data)
                print(f"✓ Fetched: {title}")
        except Exception as e:
            print(f"✗ Failed to fetch {title}: {e}")

    return articles


def analyze_correlations_from_sample(
    sample_size: int = 50,
    threshold_high: float = 0.85,
    threshold_low: float = 0.05,
) -> Dict[str, Any]:
    """Analyze correlations from a sample of articles.

    Args:
        sample_size: Number of articles to analyze.
        threshold_high: Correlation threshold for redundancy detection.
        threshold_low: Correlation threshold for weak feature detection.

    Returns:
        Dictionary containing analysis results.
    """
    print(f"\n📊 Fetching {sample_size} sample articles...")
    articles = fetch_sample_articles(sample_size)

    if not articles:
        print("❌ Failed to fetch any articles!")
        return {}

    print(f"\n✓ Successfully fetched {len(articles)} articles")

    print("\n🔍 Extracting features from articles...")
    features_list = []
    for i, article in enumerate(articles, 1):
        try:
            features = extract_all_features(article)
            if features:
                features_list.append(features)
                print(f"  [{i}/{len(articles)}] Extracted {len(features)} features")
        except Exception as e:
            print(f"  [{i}/{len(articles)}] Failed: {e}")

    if not features_list:
        print("❌ No features were extracted!")
        return {}

    print(f"\n✓ Extracted features from {len(features_list)} articles")

    # Run correlation analysis
    print(
        f"\n📈 Analyzing correlations (threshold_high={threshold_high}, threshold_low={threshold_low})..."
    )
    analyzer = CorrelationAnalyzer(
        threshold_high=threshold_high, threshold_low=threshold_low
    )
    analyzer.fit(features_list)

    report = analyzer.generate_report()
    return report


def print_report(report: Dict[str, Any]) -> None:
    """Print formatted analysis report.

    Args:
        report: Analysis report dictionary.
    """
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION ANALYSIS REPORT")
    print("=" * 80)

    print(f"\n📋 Summary:\n{report['summary']}\n")

    # High correlations
    print("\n🔴 HIGH CORRELATION PAIRS (Potential Redundancy):")
    print("-" * 80)
    if report["high_correlations"]:
        for feat1, feat2, corr in report["high_correlations"]:
            print(f"  {feat1:<30} <-> {feat2:<30} | r={corr:+.3f}")
    else:
        print("  None found! ✓")

    # Low correlations
    print("\n🟡 WEAK FEATURES (Low Correlation to Others):")
    print("-" * 80)
    if report["low_correlations"]:
        for feat, avg_corr in report["low_correlations"]:
            print(f"  {feat:<40} | avg |r|={avg_corr:.3f}")
    else:
        print("  None found! ✓")

    # Removal suggestions
    print("\n🗑️  SUGGESTED FEATURE REMOVALS:")
    print("-" * 80)
    if report["suggested_removals"]:
        for feat in report["suggested_removals"]:
            print(f"  - {feat}")
    else:
        print("  No removals suggested! ✓")

    # Multicollinearity score
    print("\n📊 MULTICOLLINEARITY SCORE:")
    print("-" * 80)
    score = report["multicollinearity_score"]
    if score < 0.3:
        status = "✓ LOW (good)"
    elif score < 0.6:
        status = "⚠️  MODERATE"
    else:
        status = "❌ HIGH (problematic)"
    print(f"  {score:.3f} {status}")

    print("\n" + "=" * 80)


def save_report(
    report: Dict[str, Any], output_file: str = "output/correlation_report.json"
) -> None:
    """Save analysis report to JSON file.

    Args:
        report: Analysis report dictionary.
        output_file: Path to output file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert lists in report to JSON-serializable format
    report_copy = report.copy()
    with open(output_path, "w") as f:
        json.dump(report_copy, f, indent=2)

    print(f"\n💾 Report saved to: {output_file}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze feature correlations in Wikipedia article dataset"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of articles to sample (default: 50)",
    )
    parser.add_argument(
        "--threshold-high",
        type=float,
        default=0.85,
        help="Correlation threshold for redundancy detection (default: 0.85)",
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=0.05,
        help="Correlation threshold for weak feature detection (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/correlation_report.json",
        help="Output file path (default: output/correlation_report.json)",
    )

    args = parser.parse_args()

    try:
        report = analyze_correlations_from_sample(
            sample_size=args.sample_size,
            threshold_high=args.threshold_high,
            threshold_low=args.threshold_low,
        )

        if report:
            print_report(report)
            save_report(report, args.output)
        else:
            print("❌ Analysis failed!")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
