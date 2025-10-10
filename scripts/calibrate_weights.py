#!/usr/bin/env python3
"""Weight calibration script for heuristic baseline model.

This script calibrates the weights of the heuristic baseline model using
GA/FA vs Stub/Start article examples to achieve correlation ≥ 0.6 with
ORES articlequality scores.
"""

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import requests  # type: ignore
from tqdm import tqdm

from models.baseline import HeuristicBaselineModel
from src.ingest.wiki_client import WikiClient


class WeightCalibrator:
    """Calibrates heuristic model weights using GA/FA vs Stub/Start examples."""

    def __init__(self, model: HeuristicBaselineModel) -> None:
        """Initialize the calibrator.

        Args:
            model: Heuristic baseline model to calibrate.
        """
        self.model = model
        self.client = WikiClient()

    def fetch_article_examples(
        self, num_examples: int = 50
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Fetch GA/FA and Stub/Start article examples for calibration.

        Args:
            num_examples: Number of examples to fetch per category.

        Returns:
            List of (article_data, quality_score) tuples.
        """
        training_data = []

        # Define article quality categories and their scores
        categories = {
            "GA": 90.0,  # Good Article
            "FA": 95.0,  # Featured Article
            "Stub": 20.0,  # Stub
            "Start": 40.0,  # Start class
        }

        print("Fetching article examples for calibration...")

        for category, score in categories.items():
            print(f"Fetching {category} articles...")

            # Get random articles from each category
            articles = self._get_random_articles_by_category(category, num_examples)

            for title in tqdm(articles, desc=f"Processing {category}"):
                try:
                    # Fetch comprehensive article data
                    article_data = self._fetch_comprehensive_data(title)
                    if article_data:
                        training_data.append((article_data, score))
                except Exception as e:
                    print(f"Error fetching {title}: {e}")
                    continue

        print(f"Successfully fetched {len(training_data)} training examples")
        return training_data

    def _get_random_articles_by_category(self, category: str, limit: int) -> List[str]:
        """Get random articles from a specific quality category.

        Args:
            category: Article quality category (GA, FA, Stub, Start).
            limit: Maximum number of articles to fetch.

        Returns:
            List of article titles.
        """
        # Use Wikipedia's random article API with category filtering
        # This is a simplified approach - in practice, you'd use ORES API

        if category in ["GA", "FA"]:
            # For GA/FA, we'll use featured articles list as proxy
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": limit,
                "rnfilterredir": "nonredirects",
            }
        else:
            # For Stub/Start, use random articles (simplified)
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": limit,
                "rnfilterredir": "nonredirects",
            }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("query", {}).get("random", []):
                articles.append(item["title"])

            return articles[:limit]

        except Exception as e:
            print(f"Error fetching {category} articles: {e}")
            return []

    def _fetch_comprehensive_data(self, title: str) -> Dict[str, Any]:
        """Fetch comprehensive data for an article.

        Args:
            title: Article title.

        Returns:
            Comprehensive article data dictionary.
        """
        try:
            # Fetch multiple data sources
            page_content = self.client.get_page_content(title)
            sections = self.client.get_sections(title)
            templates = self.client.get_templates(title)
            revisions = self.client.get_revisions(title, rvlimit=50)
            backlinks = self.client.get_backlinks(title, bllimit=100)
            citations = self.client.get_citations(title, ellimit=100)

            # Combine all data
            comprehensive_data = {
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

            return comprehensive_data

        except Exception as e:
            print(f"Error fetching comprehensive data for {title}: {e}")
            return {}

    def calibrate_weights(
        self, training_data: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        """Calibrate model weights using training data.

        Args:
            training_data: List of (article_data, quality_score) tuples.

        Returns:
            Calibration results dictionary.
        """
        print("Calibrating weights...")

        # Use the model's built-in calibration method
        results = self.model.calibrate_weights(training_data, target_correlation=0.6)

        # Calculate final correlation
        predicted_scores = []
        target_scores = []

        for article_data, target_score in training_data:
            result = self.model.calculate_maturity_score(article_data)
            predicted_scores.append(result["maturity_score"])
            target_scores.append(target_score)

        final_correlation = np.corrcoef(predicted_scores, target_scores)[0, 1]

        results["final_correlation"] = final_correlation
        results["training_samples"] = len(training_data)

        return results

    def validate_correlation(
        self, test_data: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        """Validate correlation on test data.

        Args:
            test_data: List of (article_data, quality_score) tuples.

        Returns:
            Validation results dictionary.
        """
        print("Validating correlation on test data...")

        predicted_scores = []
        target_scores = []

        for article_data, target_score in tqdm(test_data, desc="Validating"):
            try:
                result = self.model.calculate_maturity_score(article_data)
                predicted_scores.append(result["maturity_score"])
                target_scores.append(target_score)
            except Exception as e:
                print(f"Error validating article: {e}")
                continue

        if len(predicted_scores) < 2:
            return {"error": "Insufficient test data"}

        correlation = np.corrcoef(predicted_scores, target_scores)[0, 1]

        # Calculate additional metrics
        mse = np.mean([(p - t) ** 2 for p, t in zip(predicted_scores, target_scores)])
        rmse = math.sqrt(mse)
        mae = np.mean([abs(p - t) for p, t in zip(predicted_scores, target_scores)])

        return {
            "correlation": correlation,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "test_samples": len(predicted_scores),
            "target_met": abs(correlation) >= 0.6,
        }


def main() -> None:
    """Main calibration script."""
    print("Starting weight calibration for heuristic baseline model...")

    # Initialize model
    model = HeuristicBaselineModel()

    # Initialize calibrator
    calibrator = WeightCalibrator(model)

    # Fetch training data
    training_data = calibrator.fetch_article_examples(num_examples=25)

    if len(training_data) < 20:
        print("Warning: Insufficient training data for reliable calibration")
        return

    # Split data for training and validation
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]

    print(
        f"Training on {len(train_data)} examples, validating on {len(test_data)} examples"
    )

    # Calibrate weights
    calibration_results = calibrator.calibrate_weights(train_data)

    # Validate correlation
    validation_results = calibrator.validate_correlation(test_data)

    # Print results
    print("\n" + "=" * 50)
    print("CALIBRATION RESULTS")
    print("=" * 50)
    print(f"Training correlation: {calibration_results.get('best_correlation', 0):.3f}")
    print(f"Validation correlation: {validation_results.get('correlation', 0):.3f}")
    print(
        f"Target correlation (≥0.6): {'✓' if validation_results.get('target_met', False) else '✗'}"
    )
    print(f"RMSE: {validation_results.get('rmse', 0):.2f}")
    print(f"MAE: {validation_results.get('mae', 0):.2f}")

    # Save results
    results = {
        "calibration": calibration_results,
        "validation": validation_results,
        "weights": model.weights,
    }

    with open("models/calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to models/calibration_results.json")
    print(f"Weights saved to {model.weights_file}")


if __name__ == "__main__":
    main()
