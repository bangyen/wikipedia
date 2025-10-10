#!/usr/bin/env python3
"""Demo script for Wikidata completeness features.

This script demonstrates the Wikidata integration by fetching completeness
features for a sample of Wikipedia articles and showing the results.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.extractors import all_features  # noqa: E402
from features.wikidata import WikidataClient  # noqa: E402
from src.ingest.wiki_client import WikiClient  # noqa: E402


def demo_wikidata_features() -> None:
    """Demonstrate Wikidata completeness features."""
    print("🔍 Wikidata Completeness Features Demo")
    print("=" * 50)

    # Sample articles for demonstration
    sample_articles = [
        "Albert Einstein",
        "Marie Curie",
        "Leonardo da Vinci",
        "Barack Obama",
        "Galileo Galilei",
    ]

    # Initialize clients
    wiki_client = WikiClient()
    wikidata_client = WikidataClient()

    results = []

    for i, title in enumerate(sample_articles, 1):
        print(f"\n📖 {i}. Processing: {title}")
        print("-" * 30)

        try:
            # Get Wikipedia article data
            print("   🔄 Fetching Wikipedia data...")
            article_data = wiki_client.get_page_content(title)

            # Extract all features (including Wikidata)
            print("   🔄 Extracting features...")
            features = all_features(article_data)

            # Get Wikidata-specific data
            print("   🔄 Fetching Wikidata data...")
            wikidata_data = wikidata_client.get_completeness_data(title)

            # Display results
            print(f"   📊 Wikidata ID: {wikidata_data['wikidata_id']}")
            print(f"   📊 Total Statements: {wikidata_data['total_statements']}")
            print(
                f"   📊 Referenced Statements: {wikidata_data['referenced_statements']}"
            )
            print(f"   📊 Sitelinks: {wikidata_data['sitelinks_count']}")
            print(f"   📊 Claim Density: {wikidata_data['claim_density']:.2f}")
            print(f"   📊 Referenced Ratio: {wikidata_data['referenced_ratio']:.2f}")
            print(
                f"   📊 Completeness Score: {wikidata_data['completeness_score']:.3f}"
            )

            # Show extracted features
            wikidata_features = {
                k: v for k, v in features.items() if k.startswith("wikidata_")
            }
            print(f"   📊 Extracted Features: {len(wikidata_features)} features")

            # Store results
            result = {
                "title": title,
                "wikidata_data": wikidata_data,
                "wikidata_features": wikidata_features,
            }
            results.append(result)

        except Exception as e:
            print(f"   ❌ Error processing {title}: {e}")
            continue

    # Summary
    print("\n📋 SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully processed: {len(results)} articles")
    print(
        f"📊 Total Wikidata features extracted: {len(results[0]['wikidata_features']) if results else 0}"
    )

    # Feature coverage
    if results:
        has_data_count = sum(
            1 for r in results if r["wikidata_data"]["wikidata_id"] is not None
        )
        coverage = has_data_count / len(results) * 100
        print(f"📈 Wikidata coverage: {coverage:.1f}%")

        # Average metrics
        avg_statements = sum(
            r["wikidata_data"]["total_statements"] for r in results
        ) / len(results)
        avg_referenced = sum(
            r["wikidata_data"]["referenced_statements"] for r in results
        ) / len(results)
        avg_sitelinks = sum(
            r["wikidata_data"]["sitelinks_count"] for r in results
        ) / len(results)
        avg_completeness = sum(
            r["wikidata_data"]["completeness_score"] for r in results
        ) / len(results)

        print(f"📊 Average Statements: {avg_statements:.1f}")
        print(f"📊 Average Referenced: {avg_referenced:.1f}")
        print(f"📊 Average Sitelinks: {avg_sitelinks:.1f}")
        print(f"📊 Average Completeness: {avg_completeness:.3f}")

    # Save results
    output_file = project_root / "reports" / "wikidata_demo.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    # Show feature names
    if results:
        print("\n🔧 Available Wikidata Features:")
        print("-" * 30)
        feature_names = list(results[0]["wikidata_features"].keys())
        for name in sorted(feature_names):
            print(f"   • {name}")


def demo_wikidata_client() -> None:
    """Demonstrate WikidataClient functionality."""
    print("\n🔧 WikidataClient Demo")
    print("=" * 50)

    client = WikidataClient()

    # Test with Albert Einstein
    title = "Albert Einstein"
    print(f"📖 Testing with: {title}")

    try:
        # Get Wikidata ID
        print("   🔄 Getting Wikidata ID...")
        wikidata_id = client.get_wikidata_id(title)
        print(f"   📊 Wikidata ID: {wikidata_id}")

        if wikidata_id:
            # Get statements count
            print("   🔄 Getting statements count...")
            statements = client.get_statements_count(wikidata_id)
            print(f"   📊 Total Statements: {statements['total_statements']}")
            print(f"   📊 Referenced Statements: {statements['referenced_statements']}")

            # Get sitelinks count
            print("   🔄 Getting sitelinks count...")
            sitelinks = client.get_sitelinks_count(wikidata_id)
            print(f"   📊 Sitelinks: {sitelinks}")

            # Get complete data
            print("   🔄 Getting complete completeness data...")
            completeness = client.get_completeness_data(title)
            print(f"   📊 Completeness Score: {completeness['completeness_score']:.3f}")

        # Cache info
        print("   🔄 Cache information...")
        cache_info = client.get_cache_info()
        print(f"   📊 Cache Size: {cache_info['cache_size']}")
        print(f"   📊 Cache Max Size: {cache_info['cache_maxsize']}")
        print(f"   📊 Cache TTL: {cache_info['cache_ttl']}s")

    except Exception as e:
        print(f"   ❌ Error: {e}")


def main() -> None:
    """Main demo function."""
    try:
        # Demo WikidataClient
        demo_wikidata_client()

        # Demo feature extraction
        demo_wikidata_features()

        print("\n🎉 Demo completed successfully!")

    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
