"""Chat command tests using respx for HTTP mocking."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app

BASE = cfg.DEFAULT_API_URL


@pytest.fixture
def authed_profile(fake_api_key: str) -> str:
    """Save a key into the default profile so commands have something to use."""
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key, base_url=BASE))
    return fake_api_key


@respx.mock
def test_chat_one_shot_buffered(runner: CliRunner, authed_profile: str) -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "cmpl_1",
                "choices": [{"message": {"role": "assistant", "content": "hello back"}}],
            },
        )
    )
    # --json forces buffered path so the SSE branch isn't required.
    result = runner.invoke(app, ["--json", "chat", "hi"])
    assert result.exit_code == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["choices"][0]["message"]["content"] == "hello back"


@respx.mock
def test_chat_402_renders_topup_url(runner: CliRunner, authed_profile: str) -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(402, json={"error": {"message": "wallet empty"}})
    )
    result = runner.invoke(app, ["--json", "chat", "hi"])
    # Round-table lock #5 — 402 must NOT pollute stdout with half-output,
    # AND must surface a topup URL on stderr.
    assert result.exit_code == 5
    assert "Top up" in result.stderr
    assert "gpubox.ai" in result.stderr


@respx.mock
def test_chat_401_returns_auth_error(runner: CliRunner, authed_profile: str) -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    result = runner.invoke(app, ["--json", "chat", "hi"])
    assert result.exit_code == 4
    assert "auth" in result.stderr.lower()


def test_chat_without_api_key_fails_fast(runner: CliRunner) -> None:
    # No profile saved, no env var, no flag → AuthError before any network call.
    result = runner.invoke(app, ["chat", "hi"])
    assert result.exit_code == 4
    assert "API key" in result.stderr or "api key" in result.stderr.lower()


@respx.mock
def test_chat_network_error_returns_clean_exit(
    runner: CliRunner, authed_profile: str
) -> None:
    """Per Codex review: ConnectError on stream open must NOT traceback —
    it should produce a typed GPUBoxError with exit code 1 and a hint."""
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ConnectError("dns"))
    result = runner.invoke(app, ["--json", "chat", "hi"])
    assert result.exit_code == 1
    assert "could not reach" in result.stderr.lower() or "network" in result.stderr.lower()
