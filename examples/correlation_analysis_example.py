#!/usr/bin/env python3
"""Real-world example: Using correlation analysis to optimize your model.

This example shows how to:
1. Extract features from articles
2. Analyze feature correlations
3. Identify and remove redundant features
4. Recalibrate model weights with optimized features

Run with: uv run examples/correlation_analysis_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wikipedia.features.correlation_analysis import CorrelationAnalyzer  # noqa: E402
from wikipedia.models.baseline import HeuristicBaselineModel  # noqa: E402
from wikipedia.wiki_client import WikiClient  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402


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


def example_optimize_features() -> None:
    """Example: Optimize features using correlation analysis."""

    print("\n" + "=" * 80)
    print("EXAMPLE: Optimize Features Using Correlation Analysis")
    print("=" * 80)

    # Step 1: Fetch sample articles
    print("\n📚 Step 1: Fetching sample articles...")
    client = WikiClient()

    articles = []
    titles = [
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
    ]

    for title in titles[:5]:  # Use 5 articles for quick demo
        try:
            article = fetch_article_data(client, title)
            if article:
                articles.append(article)
                print(f"  ✓ {title}")
        except Exception as e:
            print(f"  ✗ {title}: {e}")

    if not articles:
        print("Failed to fetch articles!")
        return

    print(f"\n✓ Fetched {len(articles)} articles")

    # Step 2: Initialize model and extract features
    print("\n🔧 Step 2: Extracting features from articles...")
    model = HeuristicBaselineModel()

    features_list = []
    for i, article in enumerate(articles, 1):
        try:
            features = model.extract_features(article)
            features_list.append(features)
            print(f"  [{i}/{len(articles)}] Extracted {len(features)} features")
        except Exception as e:
            print(f"  [{i}/{len(articles)}] Error: {e}")

    print(f"\n✓ Extracted {len(features_list)} feature sets")
    print(f"  Total features: {len(features_list[0]) if features_list else 0}")

    # Step 3: Analyze correlations
    print("\n📊 Step 3: Analyzing feature correlations...")
    analyzer = CorrelationAnalyzer(threshold_high=0.85, threshold_low=0.05)
    analyzer.fit(features_list)

    report = analyzer.generate_report()
    print(f"\n{report['summary']}")

    # Step 4: Show redundancies
    print("\n🔴 High-Correlation Pairs (Redundancy):")
    print("-" * 80)
    high_corr = analyzer.get_high_correlations()
    if high_corr:
        for feat1, feat2, corr in high_corr[:5]:
            print(f"  {feat1:<30} <-> {feat2:<30} | r={corr:+.3f}")
        if len(high_corr) > 5:
            print(f"  ... and {len(high_corr) - 5} more")
    else:
        print("  None found!")

    # Step 5: Show weak features
    print("\n🟡 Weak Features (Low Signal):")
    print("-" * 80)
    low_corr = analyzer.get_low_correlations()
    if low_corr:
        for feat, avg_corr in low_corr[:5]:
            print(f"  {feat:<40} | avg |r|={avg_corr:.3f}")
        if len(low_corr) > 5:
            print(f"  ... and {len(low_corr) - 5} more")
    else:
        print("  None found!")

    # Step 6: Get removal suggestions
    print("\n🗑️  Suggested Feature Removals:")
    print("-" * 80)
    removals = analyzer.suggest_features_to_remove(high_corr)
    if removals:
        for feat in removals:
            print(f"  - {feat}")
        print(f"\nRemoving {len(removals)} features would reduce redundancy")
    else:
        print("  No removals suggested!")

    # Step 7: Multicollinearity assessment
    print("\n📈 Multicollinearity Score:")
    print("-" * 80)
    score = analyzer.get_multicollinearity_score()
    if score < 0.3:
        status = "✓ LOW - Good for model calibration"
    elif score < 0.6:
        status = "⚠️  MODERATE - Acceptable but monitor"
    else:
        status = "❌ HIGH - Consider removing features"

    print(f"  Score: {score:.3f}")
    print(f"  Status: {status}")

    # Step 8: Show feature correlation profiles
    print("\n📋 Feature Correlation Profiles:")
    print("-" * 80)

    interesting_features = [
        "citation_count",
        "content_length",
        "total_editors",
        "inbound_links",
    ]

    for feat in interesting_features:
        if feat in analyzer.feature_names:
            profile = analyzer.get_feature_correlation_profile(feat)
            avg_corr = sum(abs(c) for c in profile.values()) / len(profile)

            print(f"\n  {feat} (avg |r|={avg_corr:.3f}):")
            for other_feat, corr in list(profile.items())[:3]:
                print(f"    - {other_feat:<30} {corr:+.3f}")

    # Step 9: Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if score > 0.6:
        print("\n❌ High Multicollinearity Detected")
        print("   Action: Remove suggested features before calibrating weights")
        print(f"   Features to remove: {removals}")
        print("   Steps:")
        print("     1. Remove features from feature extraction")
        print("     2. Re-calibrate model weights")
        print("     3. Re-run correlation analysis")
        print("     4. Verify multicollinearity score < 0.6")

    elif score > 0.3:
        print("\n⚠️  Moderate Multicollinearity")
        print("   Status: Acceptable but could be improved")
        print("   Optional: Consider removing some redundant features")
        if removals:
            print(f"   Candidates: {removals[:3]}")

    else:
        print("\n✓ Low Multicollinearity")
        print("   Status: Good for weight calibration")
        print("   Action: Proceed with model calibration")

    # Step 10: Next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    print(
        """
1. Review the correlation analysis results above

2. Decide on feature removal strategy:
   - Option A: Keep all features (if score < 0.3)
   - Option B: Remove weakly correlated features
   - Option C: Remove all suggested redundancies (if score > 0.6)

3. Update your model:
   - Modify feature extraction to exclude removed features
   - Re-calibrate weights
   - Validate on held-out test set

4. Measure impact:
   - Compare multicollinearity scores before/after
   - Check weight stability
   - Validate correlation with ORES scores

5. Iterate:
   - Re-run correlation analysis
   - Refine feature set
   - Re-calibrate weights

For more details, see:
- features/CORRELATION_ANALYSIS.md (Full documentation)
- CORRELATION_ANALYSIS_QUICK_REFERENCE.md (Quick reference)
- scripts/analyze_correlations.py (Analysis on more articles)
"""
    )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    example_optimize_features()
