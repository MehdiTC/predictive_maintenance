"""Static contract checks for the zero-cost Render Blueprint."""

from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _blueprint() -> dict[str, Any]:
    value = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _web() -> dict[str, Any]:
    blueprint = _blueprint()
    services = {item["name"]: item for item in blueprint["services"]}
    assert set(services) == {"turbine-guard-web"}
    return cast(dict[str, Any], services["turbine-guard-web"])


def test_render_blueprint_is_one_free_web_service() -> None:
    blueprint = _blueprint()
    web = _web()
    assert web["type"] == "web"
    assert web["plan"] == "free"
    assert web["healthCheckPath"] == "/health/live"
    assert web["autoDeployTrigger"] == "checksPass"
    # Zero-cost topology: no paid MLflow service, no Render database, no disks,
    # and no paid pre-deploy phase (migrations run in the start command).
    assert "databases" not in blueprint
    assert "disk" not in web
    assert "preDeployCommand" not in web


def test_render_command_references_the_demo_entry_point() -> None:
    web = _web()
    assert web["dockerCommand"] == "python scripts/start_demo.py"
    assert (ROOT / "scripts" / "start_demo.py").is_file()
    assert (ROOT / "alembic.ini").is_file()


def test_render_env_configures_bundle_serving_without_mlflow() -> None:
    web = _web()
    env = {item["key"]: item for item in web["envVars"]}
    assert env["TURBINE_GUARD_MODEL_SOURCE"]["value"] == "deployment_bundle"
    assert not any(key.startswith("TURBINE_GUARD_MLFLOW") for key in env)
    # Account-entered values stay out of Git: the Neon URL and the pinned bundle.
    for secret_key in (
        "TURBINE_GUARD_DATABASE_URL",
        "TURBINE_GUARD_DEPLOYMENT_BUNDLE_URL",
        "TURBINE_GUARD_DEPLOYMENT_BUNDLE_SHA256",
    ):
        assert env[secret_key] == {"key": secret_key, "sync": False}
    assert env["TURBINE_GUARD_APPLICATION_SECRET"] == {
        "key": "TURBINE_GUARD_APPLICATION_SECRET",
        "generateValue": True,
    }


def test_render_demo_replay_policy_remains_bounded() -> None:
    web = _web()
    env = {item["key"]: item["value"] for item in web["envVars"] if "value" in item}
    assert env["TURBINE_GUARD_PUBLIC_DEMO_MODE"] == "true"
    assert env["TURBINE_GUARD_REPLAY_DEMO_SOURCE_ASSET_ID"] == "9"
    assert int(env["TURBINE_GUARD_REPLAY_PUBLIC_MAX_ACCELERATED_CYCLES"]) <= 10
    assert int(env["TURBINE_GUARD_REPLAY_PUBLIC_MAX_ATTEMPTS"]) <= 3


def test_render_blueprint_contains_no_literal_credentials() -> None:
    text = (ROOT / "render.yaml").read_text(encoding="utf-8").lower()
    forbidden = ("password=", "postgresql://", "api_key", "admin-token", "secret:", "neon.tech")
    assert all(value not in text for value in forbidden)
