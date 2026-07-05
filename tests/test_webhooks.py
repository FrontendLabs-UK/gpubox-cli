"""`gpb webhooks ...` command coverage (GPUB-610) — respx-mocked gateway."""

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
def test_create_defaults_to_all_training_events(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/webhooks").mock(
        return_value=httpx.Response(201, json={
            "id": "wh_1", "url": "https://r.test/h",
            "event_types": ["training.run.succeeded"], "active": True,
            "created_at": "2026-07-06T00:00:00Z", "signing_secret": "abc123",
        })
    )
    result = runner.invoke(app, ["webhooks", "create", "--url", "https://r.test/h"])
    assert result.exit_code == 0
    assert "abc123" in result.stdout  # secret shown once
    sent = json.loads(route.calls.last.request.read())
    assert sent["url"] == "https://r.test/h"
    assert set(sent["event_types"]) == {
        "training.run.created", "training.run.running", "training.run.succeeded",
        "training.run.failed", "training.run.cancelled",
    }


@respx.mock
def test_test_fire(runner: CliRunner) -> None:
    respx.post(f"{BASE}/webhooks/wh_1/test").mock(
        return_value=httpx.Response(202, json={"event_id": "test-wh_1-xyz", "event_type": "webhook.test"})
    )
    result = runner.invoke(app, ["webhooks", "test", "wh_1"])
    assert result.exit_code == 0
    assert "test-wh_1-xyz" in result.stdout


@respx.mock
def test_deliveries_lists_and_paginates(runner: CliRunner) -> None:
    route = respx.get(f"{BASE}/webhooks/wh_1/deliveries").mock(
        return_value=httpx.Response(200, json={
            "object": "list",
            "data": [
                {"event_type": "training.run.succeeded", "event_id": "trn-1-succeeded",
                 "status": "delivered", "attempt_count": 1, "next_attempt_at": None,
                 "last_attempt_error": None, "created_at": "2026-07-06T00:00:00Z",
                 "updated_at": "2026-07-06T00:00:00Z"},
            ],
            "next_cursor": "2026-07-06T00:00:00Z|abc",
        })
    )
    result = runner.invoke(app, ["webhooks", "deliveries", "wh_1", "--limit", "1"])
    assert result.exit_code == 0
    assert "training.run.succeeded" in result.stdout
    assert "delivered" in result.stdout
    assert route.calls.last.request.url.params["limit"] == "1"


@respx.mock
def test_replay(runner: CliRunner) -> None:
    respx.post(f"{BASE}/webhooks/wh_1/deliveries/trn-1-failed/replay").mock(
        return_value=httpx.Response(202, json={"status": "queued", "event_id": "trn-1-failed"})
    )
    result = runner.invoke(app, ["webhooks", "replay", "wh_1", "trn-1-failed"])
    assert result.exit_code == 0
    assert "trn-1-failed" in result.stdout


@respx.mock
def test_rotate_secret(runner: CliRunner) -> None:
    respx.post(f"{BASE}/webhooks/wh_1/rotate-secret").mock(
        return_value=httpx.Response(200, json={"id": "wh_1", "signing_secret": "newsecret999"})
    )
    result = runner.invoke(app, ["webhooks", "rotate-secret", "wh_1"])
    assert result.exit_code == 0
    assert "newsecret999" in result.stdout


@respx.mock
def test_training_presets(runner: CliRunner) -> None:
    respx.get(f"{BASE}/training/presets").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": [
            {"name": "qwen32b-lora-r16", "model_base": "Qwen/Qwen2.5-32B-Instruct-AWQ",
             "min_vram_gb": 24.0, "estimated_gpu_seconds": 28800},
        ]})
    )
    result = runner.invoke(app, ["training", "presets"])
    assert result.exit_code == 0
    assert "qwen32b-lora-r16" in result.stdout
