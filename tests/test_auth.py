"""Auth flow tests — paste-key login, status, logout."""

from __future__ import annotations

import json

import httpx
import respx
from typer.testing import CliRunner

from gpubox_cli import auth as auth_mod
from gpubox_cli import config as cfg
from gpubox_cli.main import app


def test_login_with_explicit_key_saves_profile(runner: CliRunner, fake_api_key: str) -> None:
    result = runner.invoke(app, ["auth", "login", "--api-key", fake_api_key])
    assert result.exit_code == 0, result.stderr
    profiles = cfg.load_profiles()
    assert "default" in profiles
    assert profiles["default"].api_key == fake_api_key


def test_login_rejects_short_key(runner: CliRunner) -> None:
    result = runner.invoke(app, ["auth", "login", "--api-key", "tooShort"])
    assert result.exit_code != 0


@respx.mock
def test_status_masks_key(runner: CliRunner, fake_api_key: str) -> None:
    """Per Codex review: must mock /auth/whoami to be hermetic.

    Today the gateway responds 404; we explicitly mock that so when Wave 7.5
    ships /whoami the test still works on its own data, not on the
    server's accidental behaviour.
    """
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key))
    respx.get(f"{cfg.DEFAULT_API_URL}/auth/whoami").mock(
        return_value=httpx.Response(404, json={"error": {"message": "not found"}})
    )
    result = runner.invoke(app, ["--json", "auth", "status"])
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["api_key"].startswith("gpb_test_")
    assert fake_api_key not in result.stdout, "full key must never leak"
    # Server returned 404 (endpoint not deployed) so verify_status is unverified.
    assert body["verify_status"] == "unverified"


@respx.mock
def test_status_marks_verified_when_whoami_returns_200(
    runner: CliRunner, fake_api_key: str
) -> None:
    """When Wave 7.5 ships /auth/whoami, status should mark verify=verified."""
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key))
    respx.get(f"{cfg.DEFAULT_API_URL}/auth/whoami").mock(
        return_value=httpx.Response(
            200, json={"email": "user@example.com", "tenant": "t1", "role": "admin"}
        )
    )
    result = runner.invoke(app, ["--json", "auth", "status"])
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["verify_status"] == "verified"
    assert body["server_identity"]["email"] == "user@example.com"


@respx.mock
def test_status_propagates_401(runner: CliRunner, fake_api_key: str) -> None:
    """Per Codex review: 401 must NOT silently degrade. exit code 4 + message."""
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key))
    respx.get(f"{cfg.DEFAULT_API_URL}/auth/whoami").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 4
    assert "auth" in result.stderr.lower()


def test_logout_clears_key_but_preserves_profile(
    runner: CliRunner, fake_api_key: str
) -> None:
    """Per Codex review: logout removes the secret, NOT the profile.

    base_url + default_model are user-set preferences worth keeping; the
    destructive path is `gpb profile remove`.
    """
    cfg.upsert_profile(
        "default",
        cfg.Profile(api_key=fake_api_key, base_url="https://stg.example/v1"),
    )
    result = runner.invoke(app, ["auth", "logout"])
    assert result.exit_code == 0
    profiles = cfg.load_profiles()
    assert "default" in profiles, "logout should NOT delete the profile"
    assert profiles["default"].api_key is None
    assert profiles["default"].base_url == "https://stg.example/v1"


def test_mask_key_handles_none() -> None:
    assert auth_mod.mask_key(None) == "<unset>"


def test_mask_key_short_input() -> None:
    masked = auth_mod.mask_key("abc")
    assert masked == "abc…" or masked.startswith("abc")


def test_login_non_tty_without_key_fails_cleanly(
    runner: CliRunner, monkeypatch
) -> None:
    """Per Codex review: headless login without --api-key must NOT raise
    a raw RuntimeError; it should produce a clean error + non-zero exit."""
    # Stub stdin.isatty -> False to simulate piped input.
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    result = runner.invoke(app, ["auth", "login"])
    assert result.exit_code != 0
    assert "TTY" in result.stderr or "tty" in result.stderr.lower()
