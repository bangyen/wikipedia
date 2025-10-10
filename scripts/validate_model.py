#!/usr/bin/env python3
"""Validation script for heuristic baseline model.

This script validates the heuristic baseline model by testing its correlation
with ORES articlequality scores and providing comprehensive evaluation metrics.
"""

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import requests  # type: ignore
from tqdm import tqdm

from models.baseline import HeuristicBaselineModel
from src.ingest.wiki_client import WikiClient


class ModelValidator:
    """Validates heuristic baseline model against ORES articlequality scores."""

    def __init__(self, model: HeuristicBaselineModel) -> None:
        """Initialize the validator.

        Args:
            model: Heuristic baseline model to validate.
        """
        self.model = model
        self.client = WikiClient()

    def fetch_ores_scores(self, titles: List[str]) -> Dict[str, str]:
        """Fetch ORES articlequality scores for given titles.

        Args:
            titles: List of article titles.

        Returns:
            Dictionary mapping titles to ORES quality scores.
        """
        ores_scores = {}

        # ORES API endpoint for articlequality
        url = "https://ores.wikimedia.org/v3/scores/enwiki/articlequality"

        # Process titles in batches
        batch_size = 50
        for i in tqdm(range(0, len(titles), batch_size), desc="Fetching ORES scores"):
            batch_titles = titles[i : i + batch_size]

            try:
                response = requests.post(
                    url,
                    json={"revids": [f"{title}" for title in batch_titles]},
                    timeout=30,
                )
                response.raise_for_status()

                data = response.json()

                # Extract scores from response
                for title in batch_titles:
                    if title in data.get("enwiki", {}).get("articlequality", {}):
                        score = data["enwiki"]["articlequality"][title]["score"][
                            "prediction"
                        ]
                        ores_scores[title] = score
                    else:
                        ores_scores[title] = "Unknown"

            except Exception as e:
                print(f"Error fetching ORES scores for batch: {e}")
                continue

        return ores_scores

    def get_quality_score_mapping(self) -> Dict[str, float]:
        """Get mapping from ORES quality labels to numeric scores.

        Returns:
            Dictionary mapping quality labels to numeric scores.
        """
        return {
            "FA": 95.0,  # Featured Article
            "GA": 90.0,  # Good Article
            "B": 75.0,  # B-class
            "C": 60.0,  # C-class
            "Start": 40.0,  # Start class
            "Stub": 20.0,  # Stub
            "Unknown": 50.0,  # Unknown quality
        }

    def fetch_validation_data(
        self, num_articles: int = 100
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Fetch validation data with ORES scores.

        Args:
            num_articles: Number of articles to fetch for validation.

        Returns:
            List of (article_data, ores_score) tuples.
        """
        print("Fetching validation data...")

        # Get random articles
        articles = self._get_random_articles(num_articles)

        # Fetch ORES scores
        ores_scores = self.fetch_ores_scores(articles)

        # Fetch comprehensive article data
        validation_data = []
        quality_mapping = self.get_quality_score_mapping()

        for title in tqdm(articles, desc="Processing articles"):
            try:
                # Get ORES score
                ores_label = ores_scores.get(title, "Unknown")
                ores_score = quality_mapping.get(ores_label, 50.0)

                # Fetch article data
                article_data = self._fetch_comprehensive_data(title)
                if article_data:
                    validation_data.append((article_data, ores_score))

            except Exception as e:
                print(f"Error processing {title}: {e}")
                continue

        print(f"Successfully processed {len(validation_data)} validation examples")
        return validation_data

    def _get_random_articles(self, limit: int) -> List[str]:
        """Get random articles for validation.

        Args:
            limit: Maximum number of articles to fetch.

        Returns:
            List of article titles.
        """
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
            print(f"Error fetching random articles: {e}")
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

    def validate_model(
        self, validation_data: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        """Validate the model on test data.

        Args:
            validation_data: List of (article_data, ores_score) tuples.

        Returns:
            Validation results dictionary.
        """
        print("Validating model...")

        predicted_scores = []
        target_scores = []
        pillar_scores: Dict[str, List[float]] = {
            "structure": [],
            "sourcing": [],
            "editorial": [],
            "network": [],
        }

        for article_data, target_score in tqdm(validation_data, desc="Validating"):
            try:
                result = self.model.calculate_maturity_score(article_data)
                predicted_scores.append(result["maturity_score"])
                target_scores.append(target_score)

                # Collect pillar scores for analysis
                for pillar, score in result["pillar_scores"].items():
                    pillar_scores[pillar].append(score)

            except Exception as e:
                print(f"Error validating article: {e}")
                continue

        if len(predicted_scores) < 2:
            return {"error": "Insufficient validation data"}

        # Calculate correlation
        correlation = np.corrcoef(predicted_scores, target_scores)[0, 1]

        # Calculate additional metrics
        mse = np.mean([(p - t) ** 2 for p, t in zip(predicted_scores, target_scores)])
        rmse = math.sqrt(mse)
        mae = np.mean([abs(p - t) for p, t in zip(predicted_scores, target_scores)])

        # Calculate R-squared
        ss_res = np.sum([(t - p) ** 2 for p, t in zip(predicted_scores, target_scores)])
        ss_tot = np.sum([(t - np.mean(target_scores)) ** 2 for t in target_scores])
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate pillar correlations
        pillar_correlations = {}
        for pillar, scores in pillar_scores.items():
            if len(scores) > 1:
                pillar_corr = np.corrcoef(scores, target_scores)[0, 1]
                pillar_correlations[pillar] = pillar_corr

        # Check if target correlation is met
        target_met = abs(correlation) >= 0.6

        return {
            "correlation": correlation,
            "r_squared": r_squared,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "pillar_correlations": pillar_correlations,
            "target_met": target_met,
            "validation_samples": len(predicted_scores),
            "predicted_scores": predicted_scores,
            "target_scores": target_scores,
        }

    def analyze_feature_importance(
        self, validation_data: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        """Analyze feature importance across validation data.

        Args:
            validation_data: List of (article_data, ores_score) tuples.

        Returns:
            Feature importance analysis results.
        """
        print("Analyzing feature importance...")

        feature_importance: Dict[str, List[float]] = {}
        feature_correlations: Dict[str, List[Tuple[float, float]]] = {}

        for article_data, target_score in tqdm(
            validation_data, desc="Analyzing features"
        ):
            try:
                # Get feature importance for this article
                importance = self.model.get_feature_importance(article_data)

                # Extract raw features
                raw_features = self.model.extract_features(article_data)

                # Calculate correlations
                for feature_name, value in raw_features.items():
                    if feature_name not in feature_correlations:
                        feature_correlations[feature_name] = []
                    feature_correlations[feature_name].append((value, target_score))

                # Aggregate importance
                for feature_name, imp_score in importance.items():
                    if feature_name not in feature_importance:
                        feature_importance[feature_name] = []
                    feature_importance[feature_name].append(imp_score)

            except Exception as e:
                print(f"Error analyzing features: {e}")
                continue

        # Calculate average importance and correlations
        avg_importance = {}
        for feature_name, scores in feature_importance.items():
            avg_importance[feature_name] = np.mean(scores)

        feature_corrs = {}
        for feature_name, pairs in feature_correlations.items():
            if len(pairs) > 1:
                values, targets = zip(*pairs)
                corr = np.corrcoef(values, targets)[0, 1]
                feature_corrs[feature_name] = corr

        # Sort by importance
        sorted_importance = dict(
            sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        )
        sorted_correlations = dict(
            sorted(feature_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "average_importance": sorted_importance,
            "feature_correlations": sorted_correlations,
            "top_features": list(sorted_importance.keys())[:10],
            "top_correlated": list(sorted_correlations.keys())[:10],
        }


def main() -> bool:
    """Main validation script."""
    print("Starting validation of heuristic baseline model...")

    # Initialize model
    model = HeuristicBaselineModel()

    # Initialize validator
    validator = ModelValidator(model)

    # Fetch validation data
    validation_data = validator.fetch_validation_data(num_articles=100)

    if len(validation_data) < 20:
        print("Warning: Insufficient validation data")
        return False

    # Validate model
    validation_results = validator.validate_model(validation_data)

    # Analyze feature importance
    importance_results = validator.analyze_feature_importance(validation_data)

    # Print results
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"Correlation with ORES: {validation_results.get('correlation', 0):.3f}")
    print(f"R-squared: {validation_results.get('r_squared', 0):.3f}")
    print(f"RMSE: {validation_results.get('rmse', 0):.2f}")
    print(f"MAE: {validation_results.get('mae', 0):.2f}")
    print(
        f"Target correlation (≥0.6): {'✓' if validation_results.get('target_met', False) else '✗'}"
    )

    print("\nPillar Correlations:")
    for pillar, corr in validation_results.get("pillar_correlations", {}).items():
        print(f"  {pillar}: {corr:.3f}")

    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(
        list(importance_results["average_importance"].items())[:10], 1
    ):
        print(f"  {i:2d}. {feature}: {importance:.3f}")

    print("\nTop 10 Most Correlated Features:")
    for i, (feature, corr) in enumerate(
        list(importance_results["feature_correlations"].items())[:10], 1
    ):
        print(f"  {i:2d}. {feature}: {corr:.3f}")

    # Save results
    results = {
        "validation": validation_results,
        "feature_analysis": importance_results,
        "model_info": {
            "weights_file": model.weights_file,
            "pillar_weights": model.pillar_weights,
            "feature_weights": model.feature_weights,
        },
    }

    with open("models/validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to models/validation_results.json")

    # Return success/failure
    return bool(validation_results.get("target_met", False))


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
