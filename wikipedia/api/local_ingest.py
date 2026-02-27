"""Local ingestion engine for Wikipedia data.

This module provides a client for reading Wikipedia data from local Parquet files
or HuggingFace datasets, providing a consistent interface for large-scale analysis.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd
from datasets import load_dataset  # type: ignore


class LocalIngestClient:
    """Client for reading Wikipedia data from local storage."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        hf_dataset_name: str = "wikipedia",
        hf_dataset_config: str = "20220301.en",
    ) -> None:
        """Initialize the local ingest client.

        Args:
            dataset_path: Path to local Parquet or JSONL file
            hf_dataset_name: Name of the HuggingFace dataset
            hf_dataset_config: Configuration/date for the HuggingFace dataset
        """
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.hf_dataset_name = hf_dataset_name
        self.hf_dataset_config = hf_dataset_config
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Load the dataset into memory."""
        if self.dataset_path:
            if self.dataset_path.suffix == ".parquet":
                self._df = pd.read_parquet(self.dataset_path)
            elif self.dataset_path.suffix == ".jsonl":
                self._df = pd.read_json(self.dataset_path, lines=True)
            else:
                raise ValueError(f"Unsupported file format: {self.dataset_path.suffix}")
        else:
            # Load from HuggingFace
            ds = load_dataset(
                self.hf_dataset_name,
                self.hf_dataset_config,
                split="train",
                streaming=True,
            )
            # For this MVP, we just wrap the streaming dataset or take a sample
            # In a real production scenario, we'd use the streaming API directly
            self._ds = ds

    def get_article_data(self, title: str) -> Optional[Dict[str, Any]]:
        """Get processed article data for a specific title.

        Args:
            title: Article title

        Returns:
            Dictionary in the same format as WikiClient response
        """
        if self._df is not None:
            row = self._df[self._df["title"] == title]
            if not row.empty:
                return self._format_row(row.iloc[0].to_dict())
        return None

    def _format_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a flat dataset row into the hierarchical format used by WikiClient."""
        # This is a simplified mapper
        from datetime import datetime, timezone

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": row.get("title", ""),
            "data": {
                "parse": {
                    "text": {"*": row.get("text", "")},
                },
                "query": {
                    "pages": {
                        "0": {
                            "title": row.get("title", ""),
                            "extract": row.get("text", ""),
                        }
                    }
                },
            },
        }

    def iterate_articles(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate through articles in the dataset.

        Yields:
            Article data dictionaries
        """
        if self._df is not None:
            for _, row in self._df.iterrows():
                if limit and _ >= limit:
                    break
                yield self._format_row(row.to_dict())
        elif hasattr(self, "_ds"):
            count = 0
            for row in self._ds:
                if limit and count >= limit:
                    break
                yield self._format_row(row)
                count += 1
