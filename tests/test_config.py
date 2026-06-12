"""Config layer tests — focuses on resolve() priority + perm bits."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app


def test_credentials_file_is_mode_0600(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-table lock #5: secret file must be 0600 on POSIX."""
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits don't apply on Windows")

    cfg.upsert_profile("default", cfg.Profile(api_key="gpb_test_xxxxxxxxxxxxxxxx"))
    path = cfg.credentials_path()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_settings_file_is_mode_0644(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits don't apply on Windows")
    cfg.save_settings(cfg.Settings(active_profile="dev"))
    mode = stat.S_IMODE(cfg.settings_path().stat().st_mode)
    assert mode == 0o644


def test_resolve_env_overrides_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg.upsert_profile("default", cfg.Profile(api_key="from_profile_xxxxxxxxxxx"))
    monkeypatch.setenv("GPUBOX_API_KEY", "from_env_xxxxxxxxxxxxxxxxx")
    resolved = cfg.resolve()
    assert resolved.api_key == "from_env_xxxxxxxxxxxxxxxxx"
    assert resolved.source == "env"


def test_resolve_flag_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPUBOX_API_KEY", "from_env_xxxxxxxxxxxxxxxxx")
    resolved = cfg.resolve(api_key_override="flag_xxxxxxxxxxxxxxxxxxx")
    assert resolved.api_key == "flag_xxxxxxxxxxxxxxxxxxx"
    assert resolved.source == "flag"


def test_profile_round_trips() -> None:
    cfg.upsert_profile(
        "acme", cfg.Profile(api_key="gpb_test_xxxxxxxxxxxxxxxx", base_url="https://stg.example/v1")
    )
    profiles = cfg.load_profiles()
    assert "acme" in profiles
    assert profiles["acme"].base_url == "https://stg.example/v1"


def test_remove_profile_returns_false_when_missing() -> None:
    assert cfg.remove_profile("does-not-exist") is False


def test_config_dir_respects_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-table CI-ergonomics lock: GPB_CONFIG_DIR overrides $HOME.

    Per Codex review: read paths must NOT auto-create the dir, so the
    bare config_dir() call returns the path without making it. Only
    config_dir(create=True) (used internally on save) creates the dir.
    """
    custom = tmp_path / "custom-config"
    monkeypatch.setenv("GPB_CONFIG_DIR", str(custom))
    out = cfg.config_dir()
    assert out == custom / "gpubox"
    assert not out.exists(), "read path must not create dir"
    # And the explicit create flag DOES make it.
    out2 = cfg.config_dir(create=True)
    assert out2.exists()


def test_resolve_does_not_touch_disk_when_using_env_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per Codex review: env-only invocations must work without disk writes.

    A CI runner with read-only ``$HOME`` and a GPUBOX_API_KEY env var should
    succeed at ``resolve()`` without creating any config dir on the way.
    """
    custom = tmp_path / "untouchable"
    monkeypatch.setenv("GPB_CONFIG_DIR", str(custom))
    monkeypatch.setenv("GPUBOX_API_KEY", "env_key_xxxxxxxxxxxxxxxx")
    resolved = cfg.resolve()
    assert resolved.api_key == "env_key_xxxxxxxxxxxxxxxx"
    assert resolved.source == "env"
    assert not (custom / "gpubox").exists(), "resolve() must not create dir"


# ---------------------------------------------------------------------------
# `gpb config set` secret-key guard (observation #42)
# config.toml is intentionally 0644 (world-readable). Keys that NAME a
# credential are refused (exit 2, pointer to `gpb auth login`); keys that
# merely contain a secret-ish fragment inside a longer word (tokenizer,
# keyring_backend) are stored with a stderr warning.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "API_KEY",
        "apikey",
        "apiKey",
        "auth_token",
        "TOKEN",
        "client_secret",
        "clientSecret",
        "db_password",
        "DBPassword",
        "APIToken",
        "clientIDSecret",
        "openai-key",
        "aws.secret.access",
    ],
)
def test_config_set_rejects_secret_like_keys(runner: CliRunner, key: str) -> None:
    result = runner.invoke(app, ["config", "set", key, "sk-supersecret-value"])
    assert result.exit_code == 2, result.stdout
    flat_stderr = " ".join(result.stderr.split())  # undo rich console wrapping
    assert "gpb auth login" in flat_stderr
    # Nothing persisted: neither the key nor the value may reach config.toml.
    settings = cfg.load_settings()
    assert key not in settings.extra
    path = cfg.settings_path()
    if path.exists():
        assert "sk-supersecret-value" not in path.read_text()


@pytest.mark.parametrize("key", ["tokenizer", "monkey", "keyring_backend"])
def test_config_set_fragment_only_keys_warn_but_store(runner: CliRunner, key: str) -> None:
    """Forward-compat extras stay writable when a fragment sits inside a longer word."""
    result = runner.invoke(app, ["config", "set", key, "some-plain-value"])
    assert result.exit_code == 0, result.stderr
    assert cfg.load_settings().extra[key] == "some-plain-value"
    flat_stderr = " ".join(result.stderr.split())
    assert "warning" in flat_stderr.lower()
    assert "gpb auth login" in flat_stderr


def test_config_set_legitimate_extra_key_still_works(runner: CliRunner) -> None:
    """tenant_id is the documented extras key (users.py) — must keep working."""
    result = runner.invoke(app, ["config", "set", "tenant_id", "11111111-2222-3333"])
    assert result.exit_code == 0, result.stderr
    assert cfg.load_settings().extra["tenant_id"] == "11111111-2222-3333"


def test_config_set_first_class_keys_still_work(runner: CliRunner) -> None:
    result = runner.invoke(app, ["config", "set", "default_model", "llama-3.3-70b"])
    assert result.exit_code == 0, result.stderr
    assert cfg.load_settings().default_model == "llama-3.3-70b"
