#!/usr/bin/env python3
"""Download the NASA C-MAPSS FD001 subset into the raw data layer.

Thin wrapper; all acquisition logic lives in ``turbine_guard.data``. Usage:

    uv run python scripts/download_data.py [--url URL] [--data-dir DIR] [--force]
"""

import sys

from turbine_guard.data.cli import main

if __name__ == "__main__":
    sys.exit(main())
