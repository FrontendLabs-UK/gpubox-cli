"""Embed command — happy path."""

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
def test_embed_returns_vector(runner: CliRunner) -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200, json={"data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}]}
        )
    )
    result = runner.invoke(app, ["embed", "hello world"])
    assert result.exit_code == 0
    assert "dim=8" in result.stdout
