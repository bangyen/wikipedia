"""Models package for Wikipedia article maturity scoring.

This package provides heuristic and machine learning models for assessing
Wikipedia article maturity based on structural, sourcing, editorial, and
network features.
"""

from .baseline import HeuristicBaselineModel

__all__ = ["HeuristicBaselineModel"]
