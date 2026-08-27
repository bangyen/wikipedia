"""Basic tests for the template."""

from typing import Any, Dict


def test_basic_functionality() -> None:
    """Test basic functionality."""
    assert True


def test_sample_data_fixture(sample_data: Dict[str, Any]) -> None:
    """Test that sample_data fixture works."""
    assert "name" in sample_data
    assert sample_data["name"] == "Test User"


def test_sample_config_fixture(sample_config: Dict[str, Any]) -> None:
    """Test that sample_config fixture works."""
    assert "name" in sample_config
    assert sample_config["name"] == "test-app"
