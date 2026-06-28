"""`gpb auth ...` subcommands.

Wires the auth module to the typer surface. Round-table lock #4 means
this is paste-key-only in v0.1; the ``--oidc`` flag is reserved but
errors with a clear "coming in v0.2" message so users don't think it's
broken.
"""

from __future__ import annotations

import typer

from gpubox_cli import auth
from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Authentication and identity commands.")


def _ctx(ctx: typer.Context) -> dict:
    return ctx.obj or {}


def _output(ctx: typer.Context) -> OutputCtx:
    return _ctx(ctx).get("output", OutputCtx())


def _client(ctx: typer.Context) -> GPUBoxClient:
    obj = ctx.obj or {}
    resolved = cfg.resolve(
        profile_override=obj.get("profile"),
        api_key_override=obj.get("api_key"),
        base_url_override=obj.get("base_url"),
    )
    return GPUBoxClient(ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url))


@app.command("login")
@exit_on_error
def login(
    ctx: typer.Context,
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key (paste flow). Skips interactive prompt."
    ),
    oidc: bool = typer.Option(
        False, "--oidc", help="Browser OIDC device flow (Wave 7.5; v0.2)."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Override base URL for this profile (staging/on-prem).",
    ),
) -> None:
    """Save an API key into the active profile."""
    out = _output(ctx)
    profile_name = _ctx(ctx).get("profile") or cfg.load_settings().active_profile

    if oidc:
        emit_error(out, "OIDC device flow ships in v0.2 (when Wave 7.5 deploys).")
        raise typer.Exit(2)

    key = api_key or auth.prompt_api_key_interactive()
    if not auth.looks_like_key(key):
        emit_error(out, "that doesn't look like a GPUBox API key (too short or contains whitespace)")
        raise typer.Exit(2)

    auth.save_api_key(profile_name, key, base_url=base_url)

    if out.json_mode:
        emit_json(
            out,
            {
                "ok": True,
                "profile": profile_name,
                "api_key": auth.mask_key(key),
                "base_url": base_url or cfg.DEFAULT_API_URL,
            },
        )
        return
    if not out.quiet:
        emit_text(out, f"saved key for profile '{profile_name}' ({auth.mask_key(key)})")
        emit_text(out, "tip: run `gpb auth status` to verify the key works")


@app.command("status")
@exit_on_error
def status(ctx: typer.Context) -> None:
    """Show the active identity (profile, base URL, masked key, server whoami).

    Per Codex review: exposes ``verify_status`` so users see whether the
    server actually validated the key (verified) or whether the whoami
    endpoint isn't deployed yet (unverified). 401s now raise instead of
    silently degrading.
    """
    out = _output(ctx)
    profile_name = _ctx(ctx).get("profile")
    info = auth.whoami(profile_name=profile_name)

    if out.json_mode:
        emit_json(
            out,
            {
                "profile": info.profile,
                "base_url": info.base_url,
                "api_key": info.api_key_preview,
                "source": info.source,
                "verify_status": info.verify_status,
                "server_identity": info.server_identity,
            },
        )
        return

    emit_text(out, f"profile:     {info.profile}")
    emit_text(out, f"base_url:    {info.base_url}")
    emit_text(out, f"api_key:     {info.api_key_preview}  (from {info.source})")
    emit_text(out, f"verify:      {info.verify_status}")
    if info.server_identity:
        ident = info.server_identity
        # Be defensive: the shape of /whoami may evolve as Wave 7.5 lands.
        for key in ("email", "user_id", "tenant", "role"):
            if key in ident:
                emit_text(out, f"{key:<12} {ident[key]}")


@app.command("logout")
def logout(ctx: typer.Context) -> None:
    """Clear the active profile's API key, keeping base_url + default_model.

    Per Codex review: this is the non-destructive path. To delete the
    profile entirely (including its base_url override) use
    ``gpb profile remove <name>``.
    """
    out = _output(ctx)
    profile_name = _ctx(ctx).get("profile") or cfg.load_settings().active_profile
    cleared = auth.clear_credentials(profile_name)
    if out.json_mode:
        emit_json(out, {"ok": True, "cleared": cleared, "profile": profile_name})
        return
    msg = (
        f"cleared API key for profile '{profile_name}'"
        if cleared
        else f"profile '{profile_name}' had no key to clear"
    )
    emit_text(out, msg)


@app.command("set-name", help="Set (or clear) your account display name.")
@exit_on_error
def set_name(
    ctx: typer.Context,
    display_name: str = typer.Argument(
        ...,
        help="New display name. Pass an empty string ('') to clear it.",
    ),
) -> None:
    """Update the signed-in user's display name via PATCH /v1/auth/me.

    This is a HUMAN-identity surface: the gateway rejects service (gpb_live_*)
    keys here. Use it with a user session / OIDC token. An empty/whitespace
    value clears the name (back to null).
    """
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "PATCH", "/auth/me", json_body={"display_name": display_name},
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    name = resp.get("display_name") if isinstance(resp, dict) else None
    emit_text(out, f"display name set: {name}" if name else "display name cleared")


def signup_command(
    ctx: typer.Context,
    email: str | None = typer.Option(None, "--email", help="Prefill the signup form."),
) -> None:
    """`gpb signup` — open the magic-link signup page in a browser.

    Mounted as a top-level command from main.py because it's discoverable
    that way; nothing wrong with `gpb auth signup` either, but new users
    type the obvious word first.
    """
    out = _output(ctx)
    url = auth.open_signup_browser(email)
    if out.json_mode:
        emit_json(out, {"ok": True, "url": url})
        return
    emit_text(out, f"opening {url}")
    emit_text(out, "if your browser didn't open, paste that URL manually.")
