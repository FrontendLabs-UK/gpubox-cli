"""Auth flows.

v0.1: paste-API-key only (interactive prompt or ``--api-key`` flag).
v0.2 will add OIDC device flow when Wave 7.5 deploys ACLs/SSO.

Round-table lock #4: paste-key first because it matches the dashboard
("copy this key into your client") flow — zero new concepts for users.
"""

from __future__ import annotations

import sys
import webbrowser
from dataclasses import dataclass

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient

#: API keys we issue look like ``gpb_live_<43 chars>``. We accept any
#: prefix the gateway emits (test-mode keys may use a different shape)
#: but we validate a minimum length so a typo doesn't get silently saved.
MIN_API_KEY_LENGTH = 16
SIGNUP_URL = "https://gpubox.ai/signup/"
DASHBOARD_URL = "https://gpubox.ai/dashboard/"


@dataclass
class WhoAmI:
    """Identity payload returned by /v1/auth/whoami (or whatever endpoint
    the gateway exposes — falls back gracefully when missing).

    ``verify_status`` is the explicit verdict per Codex review:
      - "unset"       → no key configured
      - "verified"    → /whoami returned 200 with this key
      - "unverified"  → /whoami endpoint not deployed (404/405); key not
                        proven valid but not proven invalid either
    """

    profile: str
    base_url: str
    api_key_preview: str  # never the full key; "gpb_live_xxx…"
    source: str  # env / profile / flag
    server_identity: dict | None = None  # populated if /whoami responds
    verify_status: str = "unset"


def mask_key(key: str | None) -> str:
    if not key:
        return "<unset>"
    if len(key) <= 12:
        # Short keys (test stubs) — show prefix only, don't risk leaking
        return key[:4] + "…"
    return key[:10] + "…" + key[-4:]


def looks_like_key(candidate: str) -> bool:
    """Cheap shape check before we save. Don't be strict: the gateway is
    the source of truth for validity, and we test live with whoami below.

    We refuse anything obviously wrong (whitespace, too short) so users
    don't paste their email by accident.
    """
    if not candidate:
        return False
    if any(ch.isspace() for ch in candidate):
        return False
    return len(candidate) >= MIN_API_KEY_LENGTH


def save_api_key(profile_name: str, api_key: str, base_url: str | None = None) -> cfg.Profile:
    """Persist the key into the named profile. Mode-0600 enforced by config layer."""
    profiles = cfg.load_profiles()
    existing = profiles.get(profile_name, cfg.Profile())
    profile = cfg.Profile(
        api_key=api_key,
        base_url=base_url or existing.base_url or cfg.DEFAULT_API_URL,
        default_model=existing.default_model,
    )
    cfg.upsert_profile(profile_name, profile)
    return profile


def clear_credentials(profile_name: str) -> bool:
    """Erase ONLY the API key on a profile, preserving base_url + default_model.

    Per Codex review: ``logout`` shouldn't drop a user's per-profile base
    URL or default model. Use ``profile remove`` for the destructive path.
    Returns True if a key was actually present (and is now cleared).
    """
    profiles = cfg.load_profiles()
    profile = profiles.get(profile_name)
    if profile is None or not profile.api_key:
        return False
    profile.api_key = None
    cfg.upsert_profile(profile_name, profile)
    return True


def clear_profile(profile_name: str) -> bool:
    """Backwards-compat shim — prefer ``clear_credentials``.

    Used by tests written against the older "logout deletes profile"
    semantics. Kept as a thin wrapper that now delegates to the
    secret-only path.
    """
    return clear_credentials(profile_name)


def open_signup_browser(email: str | None = None) -> str:
    """Best-effort browser open to the magic-link signup page.

    In headless environments (CI, SSH, no DISPLAY) ``webbrowser.open`` will
    quietly fail; we return the URL so the caller can print it instead.
    """
    url = SIGNUP_URL
    if email:
        # The signup page accepts ?email=... to prefill the input.
        from urllib.parse import urlencode

        url = f"{SIGNUP_URL}?{urlencode({'email': email})}"
    # Browser open is best-effort; suppress any failure (e.g. headless box).
    import contextlib

    with contextlib.suppress(Exception):  # pragma: no cover
        webbrowser.open_new_tab(url)
    return url


def whoami(profile_name: str | None = None) -> WhoAmI:
    """Verify the configured key against the server, with explicit verdicts.

    Per Codex review, swallowing every exception turned ``auth status``
    into a soft success even on 401/DNS failures. Now we re-raise the
    discriminating errors (auth, payment, network) so the caller can show
    the right diagnostic. Only a *missing* whoami endpoint (404 / 405)
    falls through to the "unverified" branch — because Wave 7.5 may not
    have shipped that endpoint yet.
    """
    from gpubox_cli.client import APIError, AuthError, GPUBoxError

    resolved = cfg.resolve(profile_override=profile_name)
    server_identity: dict | None = None
    verify_status: str = "unset"  # unset | verified | unverified | error

    if resolved.api_key:
        verify_status = "unverified"
        try:
            with GPUBoxClient(
                ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
            ) as client:
                server_identity = client.request("GET", "/auth/whoami")
                verify_status = "verified"
        except AuthError:
            # 401/403 → genuine bad key, surface to user.
            raise
        except APIError as exc:
            # 404/405 → endpoint not deployed yet; degrade gracefully.
            # Other 5xx → propagate so user knows server is unhealthy.
            msg = str(exc)
            if "404" in msg or "405" in msg:
                verify_status = "unverified"
            else:
                raise
        except GPUBoxError:
            # Network / timeout — re-raise so user sees the real reason.
            raise

    return WhoAmI(
        profile=resolved.profile_name,
        base_url=resolved.base_url,
        api_key_preview=mask_key(resolved.api_key),
        source=resolved.source,
        server_identity=server_identity,
        verify_status=verify_status,
    )


def prompt_api_key_interactive() -> str:
    """Read a key from a TTY, hiding it from the terminal echo.

    On non-TTY stdin this raises GPUBoxError so the @exit_on_error
    decorator on the auth login command renders a clean "use --api-key
    or GPUBOX_API_KEY" message instead of a stack trace (Codex review).
    """
    from gpubox_cli.client import GPUBoxError

    if not sys.stdin.isatty():
        raise GPUBoxError(
            "no TTY detected for interactive paste",
            hint="pass --api-key=<key>, set GPUBOX_API_KEY, or use a real terminal",
        )
    import getpass

    return getpass.getpass("Paste your GPUBox API key (input hidden): ").strip()
