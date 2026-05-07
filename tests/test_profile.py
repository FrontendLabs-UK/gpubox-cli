from __future__ import annotations

import json

from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app


def test_list_profiles_empty(runner: CliRunner) -> None:
    result = runner.invoke(app, ["profile", "list"])
    assert result.exit_code == 0
    assert "no profiles" in result.stdout


def test_use_profile_switches_active(runner: CliRunner, fake_api_key: str) -> None:
    cfg.upsert_profile("acme", cfg.Profile(api_key=fake_api_key))
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key))
    result = runner.invoke(app, ["profile", "use", "acme"])
    assert result.exit_code == 0
    assert cfg.load_settings().active_profile == "acme"


def test_use_unknown_profile_fails(runner: CliRunner) -> None:
    result = runner.invoke(app, ["profile", "use", "missing"])
    assert result.exit_code == 2


def test_remove_profile(runner: CliRunner, fake_api_key: str) -> None:
    cfg.upsert_profile("acme", cfg.Profile(api_key=fake_api_key))
    result = runner.invoke(app, ["profile", "remove", "acme", "--yes"])
    assert result.exit_code == 0
    assert "acme" not in cfg.load_profiles()


def test_list_json_includes_active(runner: CliRunner, fake_api_key: str) -> None:
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key))
    result = runner.invoke(app, ["--json", "profile", "list"])
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["active"] == "default"
    assert "default" in body["profiles"]
