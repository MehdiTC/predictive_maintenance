#!/usr/bin/env python3
"""Export or restore the immutable checksum-pinned deployment bundle."""

import sys

from turbine_guard.deployment.cli import main

if __name__ == "__main__":
    sys.exit(main())
