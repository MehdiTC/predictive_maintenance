#!/usr/bin/env python3
"""Build RUL labels, asset-level splits, and leakage-safe FD001 features.

Thin wrapper; all logic lives in ``turbine_guard.features``. Usage:

    uv run python scripts/build_features.py [--data-dir DIR] [--seed N] \
        [--rul-cap N] [--force]
"""

import sys

from turbine_guard.features.build_cli import main

if __name__ == "__main__":
    sys.exit(main())
