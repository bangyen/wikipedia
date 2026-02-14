#!/usr/bin/env python3
"""SHAP analysis for Wikipedia article maturity LightGBM model.

This script performs SHAP (SHapley Additive exPlanations) analysis on the trained
LightGBM model to understand feature importance and model interpretability.
Generates summary plots, bar charts, and detailed feature explanations.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
from wikipedia.models.train import WikipediaMaturityClassifier  # noqa: E402


class SHAPAnalyzer:
    """SHAP analysis for Wikipedia article maturity model.

    This class provides comprehensive SHAP analysis capabilities for the
    LightGBM model, including feature importance analysis, summary plots,
    and detailed explanations for model predictions.
    """

    def __init__(self, model_path: str, random_state: int = 42) -> None:
        """Initialize the SHAP analyzer.

        Args:
            model_path: Path to the trained LightGBM model.
            random_state: Random seed for reproducibility.
        """
        self.model_path = model_path
        self.random_state = random_state
        self.model: Any = None
        self.feature_names: List[str] = []
        self.explainer: Any = None
        self.shap_values: Any = None
        self.X_train: pd.DataFrame = None  # type: ignore
        self.X_test: pd.DataFrame = None  # type: ignore
        self.y_train: np.ndarray = None  # type: ignore
        self.y_test: np.ndarray = None  # type: ignore

    def load_model(self) -> None:
        """Load the trained LightGBM model and feature names."""
        print(f"Loading model from {self.model_path}...")

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]

        print(f"Model loaded successfully with {len(self.feature_names)} features")

    def prepare_data(self, sample_size: int = 500) -> None:
        """Prepare training and test data for SHAP analysis.

        Args:
            sample_size: Number of samples to generate for analysis.
        """
        print(f"Preparing dataset with {sample_size} samples...")

        # Initialize classifier to generate data
        classifier = WikipediaMaturityClassifier(random_state=self.random_state)

        # Create dataset
        X, y = classifier.create_training_dataset(
            sample_size=sample_size, ga_fa_ratio=0.3
        )

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )  # type: ignore

        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {self.X_train.shape[1]}")

    def create_explainer(self) -> None:
        """Create SHAP explainer for the LightGBM model."""
        print("Creating SHAP explainer...")

        # Use TreeExplainer for LightGBM
        self.explainer = shap.TreeExplainer(self.model)

        print("SHAP explainer created successfully")

    def compute_shap_values(self, max_samples: int = 100) -> None:
        """Compute SHAP values for the model.

        Args:
            max_samples: Maximum number of samples to compute SHAP values for.
        """
        print(f"Computing SHAP values for {max_samples} samples...")

        # Use a subset of test data for SHAP computation (can be computationally expensive)
        X_sample = self.X_test.head(max_samples)

        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X_sample)

        print("SHAP values computed successfully")

    def plot_summary(self, save_path: str = "reports/shap_summary.png") -> None:
        """Create SHAP summary plot.

        Args:
            save_path: Path to save the summary plot.
        """
        print("Creating SHAP summary plot...")

        # Use the first class SHAP values (for binary classification)
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[1]  # Use positive class
        else:
            shap_values_plot = self.shap_values

        X_sample = self.X_test.head(len(shap_values_plot))

        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_plot,
            X_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=20,
        )

        plt.title(
            "SHAP Summary Plot - Wikipedia Article Maturity Model", fontsize=16, pad=20
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Summary plot saved to {save_path}")

    def plot_bar_chart(self, save_path: str = "reports/shap_bar_chart.png") -> None:
        """Create SHAP bar chart for top features.

        Args:
            save_path: Path to save the bar chart.
        """
        print("Creating SHAP bar chart...")

        # Use the first class SHAP values (for binary classification)
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[1]  # Use positive class
        else:
            shap_values_plot = self.shap_values

        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values_plot).mean(0)

        # Get top 15 features
        top_indices = np.argsort(mean_shap)[-15:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]
        top_values = mean_shap[top_indices]

        # Create bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            range(len(top_features)), top_values, color="steelblue", alpha=0.7
        )

        # Customize plot
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel("Mean |SHAP value|", fontsize=12)
        plt.title("Top 15 Most Important Features - SHAP Values", fontsize=16, pad=20)
        plt.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_values)):
            plt.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Bar chart saved to {save_path}")

    def plot_waterfall(
        self, sample_idx: int = 0, save_path: str = "reports/shap_waterfall.png"
    ) -> None:
        """Create SHAP waterfall plot for a specific sample.

        Args:
            sample_idx: Index of the sample to explain.
            save_path: Path to save the waterfall plot.
        """
        print(f"Creating SHAP waterfall plot for sample {sample_idx}...")

        # Use the first class SHAP values (for binary classification)
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[1]  # Use positive class
        else:
            shap_values_plot = self.shap_values

        X_sample = self.X_test.head(len(shap_values_plot))

        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_plot[sample_idx],
                base_values=(
                    self.explainer.expected_value[1]
                    if isinstance(self.explainer.expected_value, list)
                    else self.explainer.expected_value
                ),
                data=X_sample.iloc[sample_idx].values,
                feature_names=self.feature_names,
            ),
            show=False,
            max_display=15,
        )

        plt.title(
            f"SHAP Waterfall Plot - Sample {sample_idx} (Mature Article Prediction)",
            fontsize=16,
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Waterfall plot saved to {save_path}")

    def create_interactive_plots(
        self, save_path: str = "reports/shap_interactive.html"
    ) -> None:
        """Create interactive SHAP plots using Plotly.

        Args:
            save_path: Path to save the interactive plot.
        """
        print("Creating interactive SHAP plots...")

        # Use the first class SHAP values (for binary classification)
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[1]  # Use positive class
        else:
            shap_values_plot = self.shap_values

        try:
            # Create interactive summary plot using Plotly backend
            import plotly.graph_objects as go  # type: ignore

            # Calculate mean absolute SHAP values for top features
            mean_abs_shap = np.abs(shap_values_plot).mean(0)
            top_indices = np.argsort(mean_abs_shap)[-15:][::-1]
            top_features = [self.feature_names[i] for i in top_indices]
            top_values = mean_abs_shap[top_indices]

            # Create interactive bar chart
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=top_values,
                        y=top_features,
                        orientation="h",
                        marker=dict(color="steelblue", opacity=0.7),
                        text=[f"{v:.3f}" for v in top_values],
                        textposition="outside",
                    )
                ]
            )

            fig.update_layout(
                title="Top 15 Most Important Features - SHAP Values",
                xaxis_title="Mean |SHAP value|",
                yaxis_title="Features",
                height=600,
                showlegend=False,
            )

            # Save as HTML
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")

        except Exception as e:
            print(f"Could not create interactive plot: {e}")
            print("Skipping interactive plot generation...")

    def generate_feature_importance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive feature importance summary.

        Returns:
            Dictionary containing feature importance analysis results.
        """
        print("Generating feature importance summary...")

        # Use the first class SHAP values (for binary classification)
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[1]  # Use positive class
        else:
            shap_values_plot = self.shap_values

        # Calculate various importance metrics
        mean_abs_shap = np.abs(shap_values_plot).mean(0)
        std_abs_shap = np.abs(shap_values_plot).std(0)
        mean_shap = shap_values_plot.mean(0)
        std_shap = shap_values_plot.std(0)

        # Create feature importance DataFrame
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs_shap,
                "std_abs_shap": std_abs_shap,
                "mean_shap": mean_shap,
                "std_shap": std_shap,
                "rank": np.argsort(mean_abs_shap)[::-1] + 1,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        # Get top 10 features
        top_10_features = importance_df.head(10)

        # Categorize features by pillar
        feature_categories = self._categorize_features()

        # Calculate pillar importance
        pillar_importance = {}
        for pillar, features in feature_categories.items():
            pillar_features = [f for f in features if f in self.feature_names]
            if pillar_features:
                feature_indices = [self.feature_names.index(f) for f in pillar_features]
                pillar_importance[pillar] = {
                    "mean_importance": mean_abs_shap[feature_indices].mean(),
                    "std_importance": mean_abs_shap[feature_indices].std(),
                    "feature_count": len(pillar_features),
                    "top_features": importance_df[
                        importance_df["feature"].isin(pillar_features)
                    ]
                    .head(3)["feature"]
                    .tolist(),
                }

        summary = {
            "model_info": {
                "model_path": self.model_path,
                "feature_count": len(self.feature_names),
                "sample_count": len(shap_values_plot),
                "expected_value": float(
                    self.explainer.expected_value[1]
                    if isinstance(self.explainer.expected_value, list)
                    else self.explainer.expected_value
                ),
            },
            "top_10_features": top_10_features.to_dict("records"),
            "pillar_importance": pillar_importance,
            "feature_categories": feature_categories,
            "all_features": importance_df.to_dict("records"),
        }

        return summary

    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features by pillar.

        Returns:
            Dictionary mapping pillar names to feature lists.
        """
        return {
            "structure": [
                "section_count",
                "content_length",
                "has_infobox",
                "template_count",
                "avg_section_depth",
                "sections_per_1k_chars",
                "has_references",
                "has_external_links",
            ],
            "sourcing": [
                "citation_count",
                "citations_per_1k_tokens",
                "external_link_count",
                "citation_density",
                "has_reliable_sources",
                "academic_source_ratio",
            ],
            "editorial": [
                "total_editors",
                "total_revisions",
                "editor_diversity",
                "recent_activity_score",
                "bot_edit_ratio",
                "major_editor_ratio",
            ],
            "network": [
                "inbound_links",
                "outbound_links",
                "connectivity_score",
                "link_density",
                "authority_score",
            ],
        }

    def save_summary_json(
        self, summary: Dict[str, Any], save_path: str = "reports/shap_summary.json"
    ) -> None:
        """Save feature importance summary to JSON file.

        Args:
            summary: Feature importance summary dictionary.
            save_path: Path to save the JSON file.
        """
        print(f"Saving summary to {save_path}...")

        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print("Summary saved successfully")

    def run_full_analysis(
        self, sample_size: int = 500, max_shap_samples: int = 100
    ) -> Dict[str, Any]:
        """Run complete SHAP analysis pipeline.

        Args:
            sample_size: Number of samples to generate for analysis.
            max_shap_samples: Maximum number of samples for SHAP computation.

        Returns:
            Feature importance summary dictionary.
        """
        print("Starting full SHAP analysis...")
        print("=" * 50)

        # Load model
        self.load_model()

        # Prepare data
        self.prepare_data(sample_size=sample_size)

        # Create explainer
        self.create_explainer()

        # Compute SHAP values
        self.compute_shap_values(max_samples=max_shap_samples)

        # Generate plots
        self.plot_summary()
        self.plot_bar_chart()
        self.plot_waterfall()
        self.create_interactive_plots()

        # Generate summary
        summary = self.generate_feature_importance_summary()

        # Save summary
        self.save_summary_json(summary)

        print("=" * 50)
        print("SHAP analysis completed successfully!")
        print("Results saved to reports/ directory")

        return summary


def main() -> bool:
    """Main function to run SHAP analysis."""
    try:
        # Path to trained model
        model_path = project_root / "models" / "gbm.pkl"

        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            print("Please train the model first using models/train.py")
            return False

        # Initialize analyzer
        analyzer = SHAPAnalyzer(str(model_path), random_state=42)

        # Run full analysis
        summary = analyzer.run_full_analysis(sample_size=500, max_shap_samples=100)

        # Print top features
        print("\nTop 10 Most Important Features:")
        print("-" * 40)
        for i, feature in enumerate(summary["top_10_features"][:10], 1):
            print(f"{i:2d}. {feature['feature']:<25} {feature['mean_abs_shap']:.4f}")

        # Print pillar importance
        print("\nPillar Importance:")
        print("-" * 40)
        for pillar, info in summary["pillar_importance"].items():
            print(
                f"{pillar.capitalize():<12} {info['mean_importance']:.4f} ± {info['std_importance']:.4f}"
            )

        return True

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
