from __future__ import annotations

import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app

BASE = cfg.DEFAULT_API_URL


@pytest.fixture(autouse=True)
def authed(fake_api_key: str) -> None:
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key, base_url=BASE))


@respx.mock
def test_promote_invokes_idempotent_post(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/hosting/models").mock(
        return_value=httpx.Response(201, json={"id": "model_xyz", "tier": "warm"})
    )
    result = runner.invoke(
        app, ["hosting", "promote", "run_abc", "--name", "acme-v1", "--tier", "warm"]
    )
    assert result.exit_code == 0
    assert "model_xyz" in result.stdout
    assert "Idempotency-Key" in route.calls.last.request.headers
    sent = json.loads(route.calls.last.request.read())
    # HostedModelCreate is extra='forbid' and requires these exact keys.
    assert sent == {
        "training_run_id": "run_abc",
        "hosted_model_name": "acme-v1",
        "hosting_tier": "warm",
    }
    # Regression: the legacy CLI keys must never be sent.
    for forbidden in ("run_id", "tier", "name"):
        assert forbidden not in sent


def test_promote_requires_name(runner: CliRunner) -> None:
    """hosted_model_name is REQUIRED server-side; --name must be mandatory so
    the CLI fails locally rather than 422-ing on an absent required field."""
    result = runner.invoke(app, ["hosting", "promote", "run_abc", "--tier", "warm"])
    assert result.exit_code != 0


@respx.mock
def test_tier_posts_to_transition_endpoint(runner: CliRunner) -> None:
    """Tier change is POST /hosting/models/{id}/transition with body
    {hosting_tier}; there is no PATCH route, and `tier` is a forbidden key."""
    route = respx.post(f"{BASE}/hosting/models/model_xyz/transition").mock(
        return_value=httpx.Response(200, json={"id": "model_xyz", "hosting_tier": "always_hot"})
    )
    result = runner.invoke(app, ["hosting", "tier", "model_xyz", "--tier", "always_hot"])
    assert result.exit_code == 0, result.stderr
    sent = json.loads(route.calls.last.request.read())
    assert sent == {"hosting_tier": "always_hot"}
    assert "tier" not in sent  # regression: legacy key must not leak


def test_invalid_tier_rejected(runner: CliRunner) -> None:
    result = runner.invoke(
        app, ["hosting", "promote", "run_a", "--name", "n", "--tier", "scorching_hot"]
    )
    assert result.exit_code != 0


@respx.mock
def test_promote_sends_workspace_header(runner: CliRunner) -> None:
    """The training run is looked up under workspace RLS — promote must send the
    same X-GPUBox-Workspace header that finetune create used, or it 404s."""
    settings = cfg.load_settings()
    settings.extra["active_workspace"] = "ws-A"
    cfg.save_settings(settings)
    route = respx.post(f"{BASE}/hosting/models").mock(
        return_value=httpx.Response(201, json={"id": "m1", "tier": "cold"})
    )
    result = runner.invoke(
        app, ["hosting", "promote", "run_abc", "--name", "m1"]
    )
    assert result.exit_code == 0, result.stderr
    assert route.calls.last.request.headers.get("X-GPUBox-Workspace") == "ws-A"


@respx.mock
def test_tier_workspace_override_wins(runner: CliRunner) -> None:
    settings = cfg.load_settings()
    settings.extra["active_workspace"] = "ws-A"
    cfg.save_settings(settings)
    route = respx.post(f"{BASE}/hosting/models/m1/transition").mock(
        return_value=httpx.Response(200, json={"id": "m1", "hosting_tier": "warm"})
    )
    result = runner.invoke(
        app, ["hosting", "tier", "m1", "--tier", "warm", "--workspace", "ws-B"]
    )
    assert result.exit_code == 0, result.stderr
    assert route.calls.last.request.headers.get("X-GPUBox-Workspace") == "ws-B"
