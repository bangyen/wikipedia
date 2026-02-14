#!/usr/bin/env python3
"""Validation script for Wikidata completeness features.

This script validates that Wikidata completeness features are available
for at least 80% of sampled Wikipedia articles, ensuring the integration
is working correctly and providing meaningful data coverage.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wikipedia.features.wikidata import wikidata_features  # noqa: E402
from wikipedia.wiki_client import WikiClient  # noqa: E402


def sample_articles(count: int = 100) -> List[str]:
    """Sample a random set of Wikipedia articles for validation.

    Args:
        count: Number of articles to sample

    Returns:
        List of article titles
    """
    # Common Wikipedia articles for testing
    sample_titles = [
        "Albert Einstein",
        "Barack Obama",
        "Marie Curie",
        "Leonardo da Vinci",
        "William Shakespeare",
        "Charles Darwin",
        "Isaac Newton",
        "Galileo Galilei",
        "Nikola Tesla",
        "Thomas Edison",
        "Alexander Graham Bell",
        "Henry Ford",
        "Wright brothers",
        "Louis Pasteur",
        "Gregor Mendel",
        "Dmitri Mendeleev",
        "Antoine Lavoisier",
        "Robert Hooke",
        "Carl Linnaeus",
        "Euclid",
        "Archimedes",
        "Pythagoras",
        "Socrates",
        "Plato",
        "Aristotle",
        "Confucius",
        "Buddha",
        "Jesus",
        "Muhammad",
        "Martin Luther",
        "John Calvin",
        "John Wesley",
        "Mother Teresa",
        "Mahatma Gandhi",
        "Nelson Mandela",
        "Martin Luther King Jr.",
        "Rosa Parks",
        "Harriet Tubman",
        "Susan B. Anthony",
        "Elizabeth Cady Stanton",
        "Eleanor Roosevelt",
        "Amelia Earhart",
        "Rosa Luxemburg",
        "Simone de Beauvoir",
        "Virginia Woolf",
        "Jane Austen",
        "Emily Dickinson",
        "Maya Angelou",
        "Toni Morrison",
        "J.K. Rowling",
        "Agatha Christie",
        "George Orwell",
        "Mark Twain",
        "Ernest Hemingway",
        "F. Scott Fitzgerald",
        "Virginia Woolf",
        "James Joyce",
        "Marcel Proust",
        "Franz Kafka",
        "Gabriel García Márquez",
        "Pablo Neruda",
        "Octavio Paz",
        "Jorge Luis Borges",
        "Gabriel García Márquez",
        "Isabel Allende",
        "Mario Vargas Llosa",
        "Carlos Fuentes",
        "Julio Cortázar",
        "Alejo Carpentier",
        "José Martí",
        "Rubén Darío",
        "José Enrique Rodó",
        "José Martí",
        "Simón Bolívar",
        "José de San Martín",
        "Bernardo O'Higgins",
        "Miguel Hidalgo",
        "José María Morelos",
        "Benito Juárez",
        "Emiliano Zapata",
        "Pancho Villa",
        "Lázaro Cárdenas",
        "Frida Kahlo",
        "Diego Rivera",
        "José Clemente Orozco",
        "David Alfaro Siqueiros",
        "Rufino Tamayo",
        "Leonora Carrington",
        "Remedios Varo",
        "Frida Kahlo",
        "Diego Rivera",
        "José Clemente Orozco",
        "David Alfaro Siqueiros",
        "Rufino Tamayo",
        "Leonora Carrington",
        "Remedios Varo",
        "Frida Kahlo",
        "Diego Rivera",
        "José Clemente Orozco",
        "David Alfaro Siqueiros",
        "Rufino Tamayo",
        "Leonora Carrington",
        "Remedios Varo",
    ]

    # Remove duplicates and sample
    unique_titles = list(set(sample_titles))
    return random.sample(unique_titles, min(count, len(unique_titles)))


def validate_wikidata_features() -> dict:
    """Validate Wikidata completeness features for sampled articles.

    Returns:
        Dictionary containing validation results
    """
    print("🔍 Validating Wikidata completeness features...")

    # Sample articles
    articles = sample_articles(50)  # Use 50 for faster validation
    print(f"📚 Sampling {len(articles)} articles for validation")

    # Initialize clients
    wiki_client = WikiClient()

    results: Dict[str, Any] = {
        "total_articles": len(articles),
        "articles_with_wikidata": 0,
        "articles_without_wikidata": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "coverage_percentage": 0.0,
        "feature_statistics": {},
        "sample_results": [],
    }

    for i, title in enumerate(articles, 1):
        print(f"📖 Processing {i}/{len(articles)}: {title}")

        try:
            # Get basic article data
            article_data = wiki_client.get_page_content(title)

            # Extract Wikidata features
            wikidata_feats = wikidata_features(article_data)

            # Check if Wikidata data is available
            has_wikidata = wikidata_feats.get("wikidata_has_data", 0.0) > 0

            if has_wikidata:
                results["articles_with_wikidata"] = (
                    int(results["articles_with_wikidata"]) + 1
                )
                results["successful_extractions"] = (
                    int(results["successful_extractions"]) + 1
                )

                # Store sample results
                sample_result = {
                    "title": title,
                    "wikidata_statements": wikidata_feats.get("wikidata_statements", 0),
                    "wikidata_referenced_statements": wikidata_feats.get(
                        "wikidata_referenced_statements", 0
                    ),
                    "wikidata_sitelinks": wikidata_feats.get("wikidata_sitelinks", 0),
                    "wikidata_claim_density": wikidata_feats.get(
                        "wikidata_claim_density", 0
                    ),
                    "wikidata_referenced_ratio": wikidata_feats.get(
                        "wikidata_referenced_ratio", 0
                    ),
                    "wikidata_completeness_score": wikidata_feats.get(
                        "wikidata_completeness_score", 0
                    ),
                }
                sample_results = results["sample_results"]
                if isinstance(sample_results, list):
                    sample_results.append(sample_result)

            else:
                results["articles_without_wikidata"] = (
                    int(results["articles_without_wikidata"]) + 1
                )
                print(f"   ⚠️  No Wikidata data found for: {title}")

        except Exception as e:
            results["failed_extractions"] = int(results["failed_extractions"]) + 1
            print(f"   ❌ Failed to process {title}: {e}")

    # Calculate coverage percentage
    articles_with_wikidata = int(results["articles_with_wikidata"])
    total_articles = int(results["total_articles"])
    results["coverage_percentage"] = articles_with_wikidata / total_articles * 100

    # Calculate feature statistics
    if results["sample_results"]:
        stats = {}
        for key in [
            "wikidata_statements",
            "wikidata_referenced_statements",
            "wikidata_sitelinks",
            "wikidata_claim_density",
            "wikidata_referenced_ratio",
            "wikidata_completeness_score",
        ]:
            sample_results = results["sample_results"]
            if isinstance(sample_results, list):
                values = [
                    r[key] for r in sample_results if isinstance(r, dict) and key in r
                ]
            else:
                values = []
            stats[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        results["feature_statistics"] = stats

    return results


def print_validation_report(results: dict) -> None:
    """Print a formatted validation report.

    Args:
        results: Validation results dictionary
    """
    print("\n" + "=" * 60)
    print("📊 WIKIDATA VALIDATION REPORT")
    print("=" * 60)

    print(f"📚 Total articles sampled: {results['total_articles']}")
    print(f"✅ Articles with Wikidata data: {results['articles_with_wikidata']}")
    print(f"❌ Articles without Wikidata data: {results['articles_without_wikidata']}")
    print(f"🔧 Successful extractions: {results['successful_extractions']}")
    print(f"💥 Failed extractions: {results['failed_extractions']}")
    print(f"📈 Coverage percentage: {results['coverage_percentage']:.1f}%")

    # Check if coverage meets requirement
    if results["coverage_percentage"] >= 80.0:
        print(
            f"\n🎉 SUCCESS: Coverage ({results['coverage_percentage']:.1f}%) meets requirement (≥80%)"
        )
    else:
        print(
            f"\n⚠️  WARNING: Coverage ({results['coverage_percentage']:.1f}%) below requirement (≥80%)"
        )

    # Feature statistics
    if results["feature_statistics"]:
        print("\n📊 FEATURE STATISTICS:")
        print("-" * 40)
        for feature, stats in results["feature_statistics"].items():
            print(f"{feature}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
            print(f"  Count: {stats['count']}")

    # Sample results
    if results["sample_results"]:
        print("\n📋 SAMPLE RESULTS (first 5):")
        print("-" * 40)
        for i, result in enumerate(results["sample_results"][:5]):
            print(f"{i+1}. {result['title']}")
            print(f"   Statements: {result['wikidata_statements']}")
            print(f"   Referenced: {result['wikidata_referenced_statements']}")
            print(f"   Sitelinks: {result['wikidata_sitelinks']}")
            print(f"   Completeness: {result['wikidata_completeness_score']:.3f}")


def main() -> None:
    """Main validation function."""
    try:
        # Run validation
        results = validate_wikidata_features()

        # Print report
        print_validation_report(results)

        # Save results to file
        output_file = project_root / "reports" / "wikidata_validation.json"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Exit with appropriate code
        if results["coverage_percentage"] >= 80.0:
            print("\n✅ Validation PASSED")
            sys.exit(0)
        else:
            print("\n❌ Validation FAILED")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
