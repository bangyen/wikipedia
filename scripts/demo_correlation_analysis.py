#!/usr/bin/env python3
"""Demo script showing feature correlation analysis in action.

This script demonstrates:
1. Creating synthetic feature data
2. Running correlation analysis
3. Interpreting results
4. Using the analysis to optimize features

Run with: uv run scripts/demo_correlation_analysis.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.correlation_analysis import CorrelationAnalyzer  # noqa: E402


def demo_basic_usage() -> None:
    """Demonstrate basic correlation analysis."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Correlation Analysis")
    print("=" * 80)

    # Create sample feature data
    features_data = [
        {
            "section_count": 5.0,
            "content_length": 1500.0,
            "citation_count": 12.0,
            "editor_count": 4.0,
        },
        {
            "section_count": 8.0,
            "content_length": 3200.0,
            "citation_count": 25.0,
            "editor_count": 8.0,
        },
        {
            "section_count": 3.0,
            "content_length": 800.0,
            "citation_count": 5.0,
            "editor_count": 2.0,
        },
        {
            "section_count": 10.0,
            "content_length": 4500.0,
            "citation_count": 35.0,
            "editor_count": 15.0,
        },
        {
            "section_count": 6.0,
            "content_length": 2000.0,
            "citation_count": 18.0,
            "editor_count": 6.0,
        },
    ]

    print("\n📊 Sample data (5 articles, 4 features):")
    for i, features in enumerate(features_data, 1):
        print(f"  Article {i}: {features}")

    # Initialize and fit analyzer
    analyzer = CorrelationAnalyzer(threshold_high=0.8, threshold_low=0.1)
    analyzer.fit(features_data)

    print("\n✓ Analyzer fitted successfully")
    print(f"  Features analyzed: {', '.join(analyzer.feature_names)}")

    # Get and display correlation matrix
    print("\n📈 Correlation Matrix:")
    if analyzer.correlation_matrix is not None:
        for i, feat_i in enumerate(analyzer.feature_names):
            print(f"  {feat_i:<20}", end="")
            for j, feat_j in enumerate(analyzer.feature_names):
                corr = analyzer.correlation_matrix[i, j]
                print(f"{corr:+.2f}  ", end="")
            print()


def demo_redundancy_detection() -> None:
    """Demonstrate redundant feature detection."""
    print("\n" + "=" * 80)
    print("DEMO 2: Redundant Feature Detection")
    print("=" * 80)

    # Create data with redundant features
    features_data = []
    for i in range(20):
        # feat_a and feat_b are highly correlated (feat_b = 2 * feat_a)
        feat_a = float(i)
        feat_b = feat_a * 2  # Perfectly correlated!

        # feat_c is independent
        feat_c = float(i % 5)

        features_data.append(
            {"citations": feat_a, "citation_density": feat_b, "editor_count": feat_c}
        )

    analyzer = CorrelationAnalyzer(threshold_high=0.9)
    analyzer.fit(features_data)

    high_corr = analyzer.get_high_correlations()

    print(f"\n🔍 Found {len(high_corr)} high-correlation pair(s):")
    for feat1, feat2, corr in high_corr:
        print(f"  {feat1} <-> {feat2}: r={corr:+.3f}")
        print("    → One of these features is likely redundant")

    print("\n🗑️  Suggested removals:")
    removals = analyzer.suggest_features_to_remove(high_corr)
    if removals:
        for feat in removals:
            print(f"  - {feat}")
    else:
        print("  None")


def demo_weak_feature_detection() -> None:
    """Demonstrate weak feature detection."""
    print("\n" + "=" * 80)
    print("DEMO 3: Weak Feature Detection")
    print("=" * 80)

    import numpy as np

    np.random.seed(42)

    # Create features where some are noise
    features_data = []
    for i in range(30):
        features_data.append(
            {
                "strong_signal": float(i),  # Strong signal
                "weak_noise": np.random.randn(),  # Random noise
                "moderate_signal": float(i * 0.5),  # Moderate correlation
            }
        )

    analyzer = CorrelationAnalyzer(threshold_low=0.2)
    analyzer.fit(features_data)

    print("\n📊 Feature Correlation Profiles:")
    for feat in analyzer.feature_names:
        profile = analyzer.get_feature_correlation_profile(feat)
        avg_corr = np.mean(np.abs(list(profile.values())))
        print(f"\n  {feat} (avg |r|={avg_corr:.3f}):")
        for other_feat, corr in list(profile.items())[:2]:
            print(f"    - {other_feat}: {corr:+.3f}")

    low_corr = analyzer.get_low_correlations()
    if low_corr:
        print("\n⚠️  Weak features (low correlation to others):")
        for feat, avg_c in low_corr:
            print(f"  - {feat}: avg |r|={avg_c:.3f}")


def demo_multicollinearity() -> None:
    """Demonstrate multicollinearity scoring."""
    print("\n" + "=" * 80)
    print("DEMO 4: Multicollinearity Assessment")
    print("=" * 80)

    # Create three datasets with different multicollinearity levels
    datasets = {
        "Independent features": [
            {"a": 1.0, "b": 10.0, "c": 100.0},
            {"a": 2.0, "b": 15.0, "c": 95.0},
            {"a": 3.0, "b": 8.0, "c": 110.0},
            {"a": 4.0, "b": 12.0, "c": 90.0},
            {"a": 5.0, "b": 18.0, "c": 105.0},
        ],
        "Moderately correlated": [
            {"x": 1.0, "y": 2.1, "z": 3.2},
            {"x": 2.0, "y": 4.1, "z": 6.1},
            {"x": 3.0, "y": 5.9, "z": 9.0},
            {"x": 4.0, "y": 8.1, "z": 11.9},
            {"x": 5.0, "y": 10.0, "z": 15.0},
        ],
        "Highly correlated": [
            {"p": 1.0, "q": 2.0, "r": 3.0},
            {"p": 2.0, "q": 4.0, "r": 6.0},
            {"p": 3.0, "q": 6.0, "r": 9.0},
            {"p": 4.0, "q": 8.0, "r": 12.0},
            {"p": 5.0, "q": 10.0, "r": 15.0},
        ],
    }

    print("\n📊 Multicollinearity Scores:")
    for name, data in datasets.items():
        analyzer = CorrelationAnalyzer()
        analyzer.fit(data)
        score = analyzer.get_multicollinearity_score()

        if score < 0.3:
            status = "✓ Low (good)"
        elif score < 0.6:
            status = "⚠️  Moderate"
        else:
            status = "❌ High (problematic)"

        print(f"  {name:<25} {score:.3f} {status}")


def demo_full_report() -> None:
    """Demonstrate full analysis report."""
    print("\n" + "=" * 80)
    print("DEMO 5: Full Analysis Report")
    print("=" * 80)

    # Create realistic feature data
    features_data = []
    for article_id in range(25):
        features_data.append(
            {
                "section_count": float(article_id % 10 + 1),
                "content_length": float((article_id + 1) * 500),
                "citation_count": float((article_id + 1) * 2),
                "external_links": float((article_id + 1) * 1.5),
                "editor_count": float(article_id % 8 + 1),
                "total_revisions": float((article_id + 1) * 5),
                "inbound_links": float((article_id + 1) * 3),
                "outbound_links": float((article_id + 1) * 2.5),
            }
        )

    analyzer = CorrelationAnalyzer(threshold_high=0.75, threshold_low=0.2)
    analyzer.fit(features_data)

    report = analyzer.generate_report()

    # Print formatted report
    print("\n" + report["summary"])

    if report["high_correlations"]:
        print("\n🔴 High Correlation Pairs:")
        for feat1, feat2, corr in report["high_correlations"][:5]:
            print(f"  {feat1} <-> {feat2}: {corr:+.3f}")

    if report["low_correlations"]:
        print("\n🟡 Weak Features:")
        for feat, corr in report["low_correlations"]:
            print(f"  {feat}: {corr:.3f}")

    if report["suggested_removals"]:
        print("\n🗑️  Suggested Removals:")
        for feat in report["suggested_removals"]:
            print(f"  - {feat}")


def main() -> None:
    """Run all demos."""
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION ANALYSIS DEMONSTRATIONS")
    print("=" * 80)

    demo_basic_usage()
    demo_redundancy_detection()
    demo_weak_feature_detection()
    demo_multicollinearity()
    demo_full_report()

    print("\n" + "=" * 80)
    print("✓ All demos completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
