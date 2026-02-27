#!/usr/bin/env python3
"""Setup script to install the wiki-score CLI tool.

This script creates a symlink or wrapper script to make the wiki-score
command available system-wide.
"""

import os
import sys
from pathlib import Path


def setup_cli() -> bool:
    """Set up the wiki-score CLI tool.

    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Get project root and CLI script path
        project_root = Path(__file__).parent.parent
        cli_script = project_root / "src" / "wikipedia" / "api" / "wiki_score.py"

        if not cli_script.exists():
            print(f"Error: CLI script not found at {cli_script}")
            return False

        # Determine installation method
        if sys.platform == "win32":
            # Windows: create batch file
            install_dir = (
                Path.home() / "AppData" / "Local" / "Microsoft" / "WindowsApps"
            )
            install_dir.mkdir(parents=True, exist_ok=True)

            batch_file = install_dir / "wiki-score.bat"
            with open(batch_file, "w") as f:
                f.write(f'@echo off\npython "{cli_script}" %*\n')

            print(f"Installed wiki-score.bat to {batch_file}")

        else:
            # Unix-like systems: create symlink or wrapper
            install_dir = Path.home() / ".local" / "bin"
            install_dir.mkdir(parents=True, exist_ok=True)

            wrapper_script = install_dir / "wiki-score"

            # Create wrapper script
            with open(wrapper_script, "w") as f:
                f.write(
                    f"""#!/bin/bash
# Wrapper script for wiki-score CLI
exec python3 "{cli_script}" "$@"
"""
                )

            # Make executable
            os.chmod(wrapper_script, 0o755)

            print(f"Installed wiki-score to {wrapper_script}")
            print("Make sure ~/.local/bin is in your PATH")

        return True

    except Exception as e:
        print(f"Error setting up CLI: {e}")
        return False


if __name__ == "__main__":
    success = setup_cli()
    sys.exit(0 if success else 1)
