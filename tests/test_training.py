"""Training submit/list happy-path."""

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
def test_submit_returns_run_id(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(200, json={"id": "run_abc"})
    )
    result = runner.invoke(
        app,
        ["training", "submit", "--preset", "deberta-base", "--dataset", "s3://bucket/x.jsonl"],
    )
    assert result.exit_code == 0
    assert "run_abc" in result.stdout
    body = route.calls.last.request.read()
    assert b"deberta-base" in body
    assert "Idempotency-Key" in route.calls.last.request.headers


@respx.mock
def test_list_runs_renders_rows(runner: CliRunner) -> None:
    respx.get(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "run_a",
                        "status": "running",
                        "preset": "deberta-base",
                        "created_at": "2026-05-07",
                    }
                ]
            },
        )
    )
    result = runner.invoke(app, ["training", "list"])
    assert result.exit_code == 0
    assert "run_a" in result.stdout
    assert "running" in result.stdout
