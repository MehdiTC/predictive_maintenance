#!/usr/bin/env python3
"""Start the free-tier public demo: restore bundle, migrate, then serve."""

import sys

from turbine_guard.deployment.startup import main

if __name__ == "__main__":
    sys.exit(main())
