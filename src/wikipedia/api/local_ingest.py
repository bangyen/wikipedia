"""Local ingestion engine for Wikipedia data.

This module provides a client for reading Wikipedia data from local Parquet files
or HuggingFace datasets, providing a consistent interface for large-scale analysis.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd
from datasets import load_dataset  # type: ignore


class LocalIngestClient:
    """Simplified client for reading raw Wikipedia data from local storage."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        hf_dataset_name: str = "wikipedia",
        hf_dataset_config: str = "20220301.en",
    ) -> None:
        """Initialize the local ingest client."""
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.hf_dataset_name = hf_dataset_name
        self.hf_dataset_config = hf_dataset_config
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Load the dataset."""
        if self.dataset_path:
            if self.dataset_path.suffix == ".parquet":
                self._df = pd.read_parquet(self.dataset_path)
            elif self.dataset_path.suffix == ".jsonl":
                self._df = pd.read_json(self.dataset_path, lines=True)
            else:
                raise ValueError(f"Unsupported: {self.dataset_path.suffix}")
        else:
            self._ds = load_dataset(
                self.hf_dataset_name,
                self.hf_dataset_config,
                split="train",
                streaming=True,
            )

    def get_article_data(self, title: str) -> Optional[Dict[str, Any]]:
        """Get raw article data (no formatting shim)."""
        if self._df is not None:
            row = self._df[self._df["title"] == title]
            if not row.empty:
                data = row.iloc[0].to_dict()
                if isinstance(data, dict):
                    return data
        return None

    def iterate_articles(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yield raw article data."""
        if self._df is not None:
            for i, (_, row) in enumerate(self._df.iterrows()):
                if limit and i >= limit:
                    break
                yield row.to_dict()
        elif hasattr(self, "_ds"):
            for i, row in enumerate(self._ds):
                if limit and i >= limit:
                    break
                yield row
