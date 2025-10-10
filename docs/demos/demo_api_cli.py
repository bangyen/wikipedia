#!/usr/bin/env python3
"""Demo script for Wikipedia maturity scoring API and CLI.

This script demonstrates both the FastAPI endpoint and CLI tool
for scoring Wikipedia articles, showing consistent results.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests  # type: ignore


def test_cli(title: str) -> dict[str, Any]:
    """Test the CLI tool.

    Args:
        title: Article title to score

    Returns:
        CLI result as dictionary
    """
    try:
        result = subprocess.run(
            [sys.executable, "cli/wiki_score.py", title, "--json"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        if result.returncode != 0:
            print(f"CLI Error: {result.stderr}")
            return {}

        return json.loads(result.stdout)  # type: ignore

    except Exception as e:
        print(f"CLI Exception: {e}")
        return {}


def test_api(title: str, base_url: str = "http://localhost:8002") -> dict[str, Any]:
    """Test the API endpoint.

    Args:
        title: Article title to score
        base_url: API base URL

    Returns:
        API result as dictionary
    """
    try:
        response = requests.get(f"{base_url}/score", params={"title": title})

        if response.status_code == 200:
            return response.json()  # type: ignore
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return {}

    except Exception as e:
        print(f"API Exception: {e}")
        return {}


def compare_results(
    cli_result: dict[str, Any], api_result: dict[str, Any], title: str
) -> bool:
    """Compare CLI and API results.

    Args:
        cli_result: CLI result dictionary
        api_result: API result dictionary
        title: Article title

    Returns:
        True if results are consistent
    """
    if not cli_result or not api_result:
        print(f"❌ Missing results for {title}")
        return False

    # Compare key fields
    fields_to_compare = ["title", "score", "band"]

    for field in fields_to_compare:
        cli_val = cli_result.get(field)
        api_val = api_result.get(field)

        if field == "score":
            # Allow small floating point differences
            if (
                cli_val is not None
                and api_val is not None
                and abs(cli_val - api_val) > 0.01
            ):
                print(f"❌ {field} mismatch: CLI={cli_val}, API={api_val}")
                return False
        else:
            if cli_val != api_val:
                print(f"❌ {field} mismatch: CLI={cli_val}, API={api_val}")
                return False

    print(f"✅ Results consistent for {title}")
    print(f"   Score: {cli_result['score']:.2f}/100 ({cli_result['band']})")
    return True


def main() -> None:
    """Main demo function."""
    print("Wikipedia Maturity Scoring Demo")
    print("=" * 50)

    # Test articles
    test_articles = [
        "Albert Einstein",
        "Python (programming language)",
        "Machine learning",
        "Wikipedia",
    ]

    # Check if API is running
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server not responding")
            print("Start the API server with: python serve/api.py")
            return
    except requests.exceptions.RequestException:
        print("❌ API server not running")
        print("Start the API server with: python serve/api.py")
        return

    print("✅ API server is running")
    print()

    # Test each article
    consistent_count = 0

    for title in test_articles:
        print(f"Testing: {title}")
        print("-" * 30)

        # Test CLI
        print("Testing CLI...")
        cli_result = test_cli(title)

        # Test API
        print("Testing API...")
        api_result = test_api(title)

        # Compare results
        if compare_results(cli_result, api_result, title):
            consistent_count += 1

        print()

    # Summary
    print("=" * 50)
    print(f"Summary: {consistent_count}/{len(test_articles)} tests passed")

    if consistent_count == len(test_articles):
        print("🎉 All tests passed! CLI and API are consistent.")
    else:
        print("⚠️  Some tests failed. Check the differences above.")

    # Show CLI formatted output
    print("\n" + "=" * 50)
    print("CLI Formatted Output Example:")
    print("=" * 50)

    try:
        subprocess.run(
            [sys.executable, "cli/wiki_score.py", "Albert Einstein"],
            cwd=Path(__file__).parent,
        )
    except Exception as e:
        print(f"Error showing CLI output: {e}")


if __name__ == "__main__":
    main()
