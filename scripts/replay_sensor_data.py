#!/usr/bin/env python3
"""Replay held-out sensor trajectories through the running TurbineGuard API.

Thin wrapper; all logic lives in ``turbine_guard.replay``. Usage:

    uv run python scripts/replay_sensor_data.py start --asset-id 9
    uv run python scripts/replay_sensor_data.py step --run-id <UUID>
    uv run python scripts/replay_sensor_data.py resume --run-id <UUID>
    uv run python scripts/replay_sensor_data.py status --run-id <UUID>
    uv run python scripts/replay_sensor_data.py stop --run-id <UUID>
    uv run python scripts/replay_sensor_data.py evaluate-aggregate
"""

import sys

from turbine_guard.replay.cli import main

if __name__ == "__main__":
    sys.exit(main())
