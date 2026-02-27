"""Bulk processing script for large-scale Wikipedia article analysis.

This script demonstrates the integration of LocalIngestClient (Phase 2) and
GraphProcessor (Phase 3) to generate features for thousands of articles
using high-fidelity global metrics.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wikipedia.api.local_ingest import LocalIngestClient  # noqa: E402
from wikipedia.features.extractors import all_features  # noqa: E402
from wikipedia.features.graph_processor import GraphProcessor  # noqa: E402


def bulk_process(
    dataset_path: Optional[str] = None,
    links_path: Optional[str] = None,
    output_path: str = "bulk_features.parquet",
    sample_size: int = 100,
) -> pd.DataFrame:
    """Run the full bulk processing pipeline.

    Args:
        dataset_path: Path to local article dataset (Parquet/JSONL)
        links_path: Path to local link data (Parquet/CSV)
        output_path: Path to save result
        sample_size: Number of articles to process

    Returns:
        DataFrame containing generated features
    """
    print("Step 1: Initializing Ingestion & Graph Analysis")
    ingest_client = LocalIngestClient(dataset_path=dataset_path)
    graph_processor = GraphProcessor()

    # Phase 3: Global Graph Ingestion (if links provided)
    if links_path:
        print(f"Ingesting global links from {links_path}...")
        graph_processor.ingest_links(links_path)
        G = graph_processor.build_networkx_graph()
        metrics = graph_processor.compute_global_metrics(G)
        graph_processor.save_metrics_to_db(metrics)
        print("Global metrics computed and cached.")

    # Phase 2: Bulk Ingestion
    print("Loading article dataset...")
    ingest_client.load()

    all_rows = []
    print(f"Processing up to {sample_size} articles...")

    for article_data in tqdm(
        ingest_client.iterate_articles(limit=sample_size), total=sample_size
    ):
        try:
            # Generate features using global graph metrics from graph_processor
            features = all_features(article_data, graph_processor=graph_processor)

            # Include metadata
            features["title"] = article_data.get("title", "Unknown")
            all_rows.append(features)
        except Exception as e:
            print(f"Error processing {article_data.get('title')}: {e}")
            continue

    # Create result DataFrame
    df = pd.DataFrame(all_rows)
    df.to_parquet(output_path, index=False)
    print(f"\nBulk processing complete! Saved {len(df)} articles to {output_path}")

    return df


if __name__ == "__main__":
    # Example usage (can be customized by user)
    print("Wikipedia Full-Scale Data Pipeline Demonstration")
    print("=" * 50)

    # In a real scenario, the user would provide paths to large dumps
    # For this demo, we can run on a small sample if provided
    # bulk_process(sample_size=10)
    print("Run this script with local data paths to begin large-scale ingestion.")
