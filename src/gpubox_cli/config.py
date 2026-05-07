"""Config + credential storage.

Round-table lock #5: file-based, mode 0600, no system keyring in v0.1.
File-based with strict perms is the kubectl/gh/aws-cli baseline. Headless
boxes (CI runners, remote shells) make keyring support painful, so we defer
that to v0.2 as an explicit opt-in backend.

Round-table lock (CI ergonomics): respect ``GPB_CONFIG_DIR`` so headless
runners with no ``$HOME`` (or a read-only one) can point us at /tmp.

Round-table lock #10: profiles are first-class from day 1. Multiple keys,
multiple base URLs (staging, prod, future on-prem) — all selected via the
``--profile`` global flag or ``GPB_PROFILE`` env var.

We split *config* (non-secret defaults: base URL, default model, profile
preference) from *credentials* (the API keys) per Codex's nudge — the
credentials file gets 0600 perms, the config file gets 0644, and they age
better as separate files.
"""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

DEFAULT_API_URL = "https://api.gpubox.ai/v1"
DEFAULT_PROFILE = "default"

#: Names of env vars consulted during credential lookup, in priority order.
ENV_API_KEY = "GPUBOX_API_KEY"
ENV_API_URL = "GPUBOX_API_URL"
ENV_PROFILE = "GPB_PROFILE"
ENV_CONFIG_DIR = "GPB_CONFIG_DIR"
ENV_NO_COLOR = "NO_COLOR"


def _xdg_config_home() -> Path:
    """Return the XDG-spec config root, with platform fallbacks.

    Priority:
      1. ``GPB_CONFIG_DIR`` if set (CI override per round-table lock)
      2. ``XDG_CONFIG_HOME`` if set (Linux + most BSDs)
      3. ``~/Library/Application Support`` on macOS
      4. ``%APPDATA%`` on Windows
      5. ``~/.config`` everywhere else
    """
    override = os.environ.get(ENV_CONFIG_DIR)
    if override:
        return Path(override).expanduser()

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata)
    return Path.home() / ".config"


def config_dir(*, create: bool = False) -> Path:
    """Return the gpubox-specific config directory.

    Per Codex review: read paths must NOT touch the filesystem. Pure
    env-only / flag-only invocations (``gpb --api-key X chat ...``) on a
    box with no writable HOME should still work. Only opt into directory
    creation when we're actually about to save (set ``create=True``).
    """
    root = _xdg_config_home() / "gpubox"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def credentials_path() -> Path:
    return config_dir() / "credentials.toml"


def settings_path() -> Path:
    return config_dir() / "config.toml"


@dataclass
class Profile:
    """One named credential bundle. Keys are ALWAYS stored in credentials.toml.

    base_url is allowed to override the global default per-profile so a
    consultant can point one profile at staging without touching the rest.
    """

    api_key: str | None = None
    base_url: str = DEFAULT_API_URL
    default_model: str | None = None


@dataclass
class Settings:
    """Non-secret defaults. Stored in config.toml at 0644."""

    active_profile: str = DEFAULT_PROFILE
    default_model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _write_toml(path: Path, data: dict[str, Any], *, secret: bool) -> None:
    """Atomic write with the right permission bits.

    secret=True -> 0600 (owner read/write only). Otherwise 0644.
    Atomic via NamedTemporaryFile + os.replace so a crash mid-write
    doesn't corrupt the existing file. This is the *only* path that
    creates the config directory — read paths stay disk-free.
    """
    # config_dir(create=True) here ensures the parent exists exactly once,
    # at the moment we're about to write. Per Codex review.
    config_dir(create=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = tomli_w.dumps(data).encode("utf-8")
    with tmp.open("wb") as fh:
        fh.write(payload)
    if sys.platform != "win32":
        # Set perms BEFORE the atomic replace so there's no readable window.
        mode = stat.S_IRUSR | stat.S_IWUSR if secret else (
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        )
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def load_settings() -> Settings:
    raw = _read_toml(settings_path())
    return Settings(
        active_profile=raw.get("active_profile", DEFAULT_PROFILE),
        default_model=raw.get("default_model"),
        extra={k: v for k, v in raw.items() if k not in {"active_profile", "default_model"}},
    )


def save_settings(settings: Settings) -> None:
    payload: dict[str, Any] = {"active_profile": settings.active_profile}
    if settings.default_model:
        payload["default_model"] = settings.default_model
    payload.update(settings.extra)
    _write_toml(settings_path(), payload, secret=False)


def load_profiles() -> dict[str, Profile]:
    raw = _read_toml(credentials_path())
    profiles_raw = raw.get("profiles", {})
    out: dict[str, Profile] = {}
    for name, data in profiles_raw.items():
        if not isinstance(data, dict):
            continue
        out[name] = Profile(
            api_key=data.get("api_key"),
            base_url=data.get("base_url", DEFAULT_API_URL),
            default_model=data.get("default_model"),
        )
    return out


def save_profiles(profiles: dict[str, Profile]) -> None:
    """Write all profiles to credentials.toml at 0600."""
    payload: dict[str, Any] = {"profiles": {}}
    for name, profile in profiles.items():
        entry: dict[str, Any] = {"base_url": profile.base_url}
        if profile.api_key:
            entry["api_key"] = profile.api_key
        if profile.default_model:
            entry["default_model"] = profile.default_model
        payload["profiles"][name] = entry
    _write_toml(credentials_path(), payload, secret=True)


def upsert_profile(name: str, profile: Profile) -> None:
    profiles = load_profiles()
    profiles[name] = profile
    save_profiles(profiles)


def remove_profile(name: str) -> bool:
    profiles = load_profiles()
    if name not in profiles:
        return False
    del profiles[name]
    save_profiles(profiles)
    return True


@dataclass
class ResolvedConfig:
    """Final config used by a command, after env-var + profile + flag merge."""

    api_key: str | None
    base_url: str
    profile_name: str
    default_model: str | None
    source: str  # "env" | "profile" | "flag"


def resolve(
    *,
    profile_override: str | None = None,
    api_key_override: str | None = None,
    base_url_override: str | None = None,
) -> ResolvedConfig:
    """Compute the effective config for a command invocation.

    Priority (highest first):
      1. Explicit ``--api-key`` / ``--base-url`` flag overrides
      2. ``GPUBOX_API_KEY`` / ``GPUBOX_API_URL`` env vars
      3. The named profile from ``--profile`` or ``GPB_PROFILE``
      4. The active profile from config.toml
      5. The hardcoded default profile

    This ordering means a CI run can set ``GPUBOX_API_KEY`` without touching
    files, and a local dev can flip profiles without exporting anything.
    """
    settings = load_settings()
    profiles = load_profiles()

    profile_name = profile_override or os.environ.get(ENV_PROFILE) or settings.active_profile
    profile = profiles.get(profile_name, Profile())

    env_key = os.environ.get(ENV_API_KEY)
    env_url = os.environ.get(ENV_API_URL)

    if api_key_override:
        api_key, source = api_key_override, "flag"
    elif env_key:
        api_key, source = env_key, "env"
    else:
        api_key, source = profile.api_key, "profile"

    base_url = base_url_override or env_url or profile.base_url or DEFAULT_API_URL

    return ResolvedConfig(
        api_key=api_key,
        base_url=base_url,
        profile_name=profile_name,
        default_model=profile.default_model or settings.default_model,
        source=source,
    )
