#!/usr/bin/env python3
"""Process the acquired NASA C-MAPSS FD001 raw files into validated Parquet.

Thin wrapper; all processing logic lives in ``turbine_guard.data``. Usage:

    uv run python scripts/process_data.py [--data-dir DIR] [--force]
"""

import sys

from turbine_guard.data.process_cli import main

if __name__ == "__main__":
    sys.exit(main())
