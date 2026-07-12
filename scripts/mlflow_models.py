#!/usr/bin/env python3
"""Inspect and verify TurbineGuard MLflow runs and registered models."""

import sys

from turbine_guard.tracking.cli import main

if __name__ == "__main__":
    sys.exit(main())
