from __future__ import annotations

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
        return_value=httpx.Response(200, json={"id": "model_xyz", "tier": "warm"})
    )
    result = runner.invoke(app, ["hosting", "promote", "run_abc", "--tier", "warm"])
    assert result.exit_code == 0
    assert "model_xyz" in result.stdout
    assert "Idempotency-Key" in route.calls.last.request.headers


def test_invalid_tier_rejected(runner: CliRunner) -> None:
    result = runner.invoke(app, ["hosting", "promote", "run_a", "--tier", "scorching_hot"])
    assert result.exit_code != 0
