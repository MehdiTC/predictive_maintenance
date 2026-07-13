#!/usr/bin/env python3
"""Small stdlib HTTP health probe used by Compose without curl."""

import argparse
import json
import os
import urllib.request
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="healthcheck")
    parser.add_argument("service", choices=("api", "live", "mlflow"))
    args = parser.parse_args(argv)
    if args.service in ("api", "live"):
        port = int(os.getenv("TURBINE_GUARD_API_PORT", "8000"))
        path = "ready" if args.service == "api" else "live"
        url = f"http://127.0.0.1:{port}/health/{path}"
    else:
        url = "http://127.0.0.1:5000/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status != 200:
                return 1
            if args.service == "api":
                payload = json.load(response)
                checks = payload.get("checks", {})
                if payload.get("status") != "ready" or not checks or not all(checks.values()):
                    return 1
    except (OSError, ValueError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
