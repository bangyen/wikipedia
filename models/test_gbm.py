#!/usr/bin/env python3
"""Test script for the trained LightGBM model.

This script loads the trained model and tests it on sample articles
to verify it works correctly.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
from models.train import WikipediaMaturityClassifier  # noqa: E402
from src.ingest.wiki_client import WikiClient  # noqa: E402


def test_model() -> None:
    """Test the trained LightGBM model on sample articles."""
    print("Testing LightGBM Model")
    print("=" * 30)

    # Load the trained model
    classifier = WikipediaMaturityClassifier()
    model_path = Path(__file__).parent / "gbm.pkl"

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print("Please run models/train.py first to train the model.")
        return

    classifier.load_model(str(model_path))
    print(f"Model loaded from {model_path}")

    # Test articles
    test_articles = [
        "Albert Einstein",  # Should be high quality (GA/FA)
        "Python (programming language)",  # Should be high quality
        "Stub",  # Should be low quality
        "Wikipedia",  # Should be high quality
    ]

    client = WikiClient()

    for title in test_articles:
        print(f"\nTesting: {title}")

        try:
            # Fetch article data
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

            # Extract features
            features = classifier.extract_all_features(article_data)

            # Convert to DataFrame
            features_df = pd.DataFrame([features])

            # Make prediction
            prediction_proba = classifier.predict(features_df)[0]
            prediction = 1 if prediction_proba > 0.5 else 0

            # Display results
            quality_class = "GA/FA" if prediction == 1 else "Other"
            confidence = max(prediction_proba, 1 - prediction_proba)

            print(f"  Prediction: {quality_class}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Probability: {prediction_proba:.3f}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\nModel test completed!")


if __name__ == "__main__":
    test_model()
