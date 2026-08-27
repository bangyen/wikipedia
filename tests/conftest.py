"""
Pytest configuration and shared fixtures.
"""

from typing import Any, Dict

import pytest


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Provide sample data for tests."""
    return {"name": "Test User", "age": 30}


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for tests."""
    return {"name": "test-app", "version": "1.0.0"}
