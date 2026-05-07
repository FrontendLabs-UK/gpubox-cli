"""Shared fixtures.

Two big jobs:

1. Force ``GPB_CONFIG_DIR`` to a per-test tmp dir so we never touch the
   real user's credentials.toml during tests. This also exercises the
   round-table CI-ergonomics lock — config writes work without ``$HOME``.
2. Provide a typer CliRunner factory and a respx-mocked HTTP transport.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point GPB_CONFIG_DIR at a tmp dir for every test.

    autouse=True so we never accidentally pollute ~/.config/gpubox during
    a test run. Also clears env vars that would short-circuit our resolver.
    """
    cfg_dir = tmp_path / "gpb-cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("GPB_CONFIG_DIR", str(cfg_dir))
    # Wipe any inherited gpubox env so tests are deterministic.
    for var in ("GPUBOX_API_KEY", "GPUBOX_API_URL", "GPB_PROFILE", "NO_COLOR"):
        monkeypatch.delenv(var, raising=False)
    return cfg_dir


@pytest.fixture
def runner() -> CliRunner:
    # Newer typer/click separate stderr by default; older accepted
    # mix_stderr=False explicitly. Try the kwarg first, fall back silently.
    try:
        return CliRunner(mix_stderr=False)  # type: ignore[call-arg]
    except TypeError:
        return CliRunner()


@pytest.fixture
def fake_api_key() -> str:
    """A key that passes our shape check but isn't a real prod key."""
    return "gpb_test_" + ("a" * 40)
