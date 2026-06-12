"""`gpb config get/set` — read/write non-secret settings.

Stored in config.toml (NOT credentials.toml) so this is safe to inspect
casually. Currently supported keys:

* ``active_profile``  — name of the default profile
* ``default_model``   — model id used when --model isn't passed

Other keys are tucked into ``settings.extra`` for forward compatibility;
unknown keys round-trip safely. Exception: key names that name a credential
outright (api_key, auth_token, client_secret, password, ...) are refused,
because config.toml is world-readable — `gpb auth login` is the secrets
path. Names that merely contain a secret-ish fragment (tokenizer,
keyring_backend) are stored with a stderr warning.
"""

from __future__ import annotations

import re

import typer

from gpubox_cli import config as cfg
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text, emit_warning

app = typer.Typer(no_args_is_help=True, help="Read / write CLI config.")

# Allowlist of "first-class" config keys. Unknown keys still work via
# settings.extra but won't render through `gpb config get` without a key arg.
_KNOWN_KEYS = {"active_profile", "default_model"}

# Secret-looking key names are refused outright: config.toml is intentionally
# world-readable (0644), so `gpb config set api_key ...` would write a secret
# where every local user can read it. Credentials belong in credentials.toml
# (0600) via `gpb auth login`.
#
# Two-tier match (Codex review: extras are a documented forward-compat
# surface, so a blunt substring hard-block would also kill innocent keys
# like `tokenizer`, `monkey`, or a future v0.2 `keyring_backend`):
#
# * HARD BLOCK when a whole word of the key (split on separators and
#   camelCase) is a credential noun — api_key, authToken, client-secret,
#   APIKEY, db_password all land here. Exit 2.
# * WARN-ONLY when a fragment merely appears inside a longer word
#   (tokenizer, keyring_backend): stored as requested, with a stderr nudge
#   towards `gpb auth login` in case it really is a credential.
#
# None of today's legitimate keys (active_profile, default_model,
# tenant_id) trip either tier.
_SECRET_KEY_TOKENS = {
    "key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "credentials",
}
_SECRET_KEY_FRAGMENTS = ("key", "token", "secret", "password")


def _key_words(key: str) -> list[str]:
    """Split a key name into lowercase words on separators + camelCase.

    The second alternative handles acronym boundaries (DBPassword, APIToken,
    clientIDSecret) — an upper run followed by Upper+lower starts a new word.
    """
    decamelled = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", key)
    return [word for word in re.split(r"[^a-zA-Z0-9]+", decamelled.lower()) if word]


def _secret_token(key: str) -> str | None:
    """Return the credential noun when *key* names a secret outright."""
    for word in _key_words(key):
        if word in _SECRET_KEY_TOKENS:
            return word
    return None


def _secret_fragment(key: str) -> str | None:
    """Return the matched fragment when *key* merely smells like a secret."""
    lowered = key.lower()
    for fragment in _SECRET_KEY_FRAGMENTS:
        if fragment in lowered:
            return fragment
    return None


def _output(ctx: typer.Context) -> OutputCtx:
    return (ctx.obj or {}).get("output", OutputCtx())


def _flatten(settings: cfg.Settings) -> dict[str, object]:
    return {
        "active_profile": settings.active_profile,
        "default_model": settings.default_model,
        **settings.extra,
    }


@app.command("get")
def get_value(ctx: typer.Context, key: str | None = typer.Argument(None)) -> None:
    """Print one config value, or all of them when no key is given."""
    out = _output(ctx)
    settings = cfg.load_settings()
    flat = _flatten(settings)

    if key:
        if key not in flat:
            emit_error(out, f"no such config key: {key}")
            raise typer.Exit(2)
        if out.json_mode:
            emit_json(out, {key: flat[key]})
        else:
            emit_text(out, str(flat[key]) if flat[key] is not None else "")
        return

    if out.json_mode:
        emit_json(out, flat)
        return
    for k in sorted(flat):
        emit_text(out, f"{k}={flat[k] if flat[k] is not None else ''}")


@app.command("set")
def set_value(
    ctx: typer.Context,
    key: str = typer.Argument(...),
    value: str = typer.Argument(...),
) -> None:
    """Set one config value. Use `gpb profile use <name>` for active profile."""
    out = _output(ctx)
    token = _secret_token(key)
    if token is not None:
        emit_error(
            out,
            f"refusing to store '{key}' (names a '{token}'): config.toml "
            "is world-readable (0644), so secrets don't belong there. "
            "run `gpb auth login` to store credentials safely (0600).",
        )
        raise typer.Exit(2)
    if _secret_fragment(key) is not None:
        emit_warning(
            out,
            f"'{key}' looks like it could name a credential. config.toml is "
            "world-readable (0644) — if this value is a secret, remove it and "
            "use `gpb auth login` instead.",
        )
    settings = cfg.load_settings()
    if key == "active_profile":
        settings.active_profile = value
    elif key == "default_model":
        settings.default_model = value or None
    else:
        # Unknown key — store as extras. Doesn't break anyone.
        settings.extra[key] = value
    cfg.save_settings(settings)
    if out.json_mode:
        emit_json(out, {"ok": True, "key": key, "value": value})
        return
    emit_text(out, f"set {key}={value}")
