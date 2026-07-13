#!/usr/bin/env python3
"""Manage monitoring, retraining candidates, promotion, refresh, and rollback."""

import sys

from turbine_guard.monitoring.cli import main

if __name__ == "__main__":
    sys.exit(main())
