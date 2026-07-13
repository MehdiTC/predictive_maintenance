#!/usr/bin/env python3
"""Start the production-style TurbineGuard API process."""

import sys

from turbine_guard.api.cli import main

if __name__ == "__main__":
    sys.exit(main())
