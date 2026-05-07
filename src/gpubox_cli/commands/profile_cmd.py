"""`gpb profile ...` — list / add / remove / use profiles.

Per Codex's nudge in round 1: profiles deserve a top-level noun, not just
a `--profile` flag. This module exposes the verbs.
"""

from __future__ import annotations

import typer

from gpubox_cli import config as cfg
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Manage credential profiles.")


def _output(ctx: typer.Context) -> OutputCtx:
    return (ctx.obj or {}).get("output", OutputCtx())


@app.command("list")
def list_profiles(ctx: typer.Context) -> None:
    """List configured profiles + which is active."""
    out = _output(ctx)
    profiles = cfg.load_profiles()
    settings = cfg.load_settings()

    if out.json_mode:
        emit_json(
            out,
            {
                "active": settings.active_profile,
                "profiles": {
                    name: {
                        "base_url": p.base_url,
                        "default_model": p.default_model,
                        "has_key": bool(p.api_key),
                    }
                    for name, p in profiles.items()
                },
            },
        )
        return

    if not profiles:
        emit_text(out, "no profiles configured. run `gpb auth login` to add one.")
        return
    for name, profile in sorted(profiles.items()):
        marker = "*" if name == settings.active_profile else " "
        key_state = "set" if profile.api_key else "missing"
        emit_text(out, f"{marker} {name:<16} {profile.base_url}  (key: {key_state})")


@app.command("use")
def use_profile(ctx: typer.Context, name: str = typer.Argument(...)) -> None:
    """Set the active profile (used when --profile isn't passed)."""
    out = _output(ctx)
    profiles = cfg.load_profiles()
    if name not in profiles:
        emit_error(out, f"no profile named '{name}'. run `gpb auth login --profile {name}` first.")
        raise typer.Exit(2)
    settings = cfg.load_settings()
    settings.active_profile = name
    cfg.save_settings(settings)
    if out.json_mode:
        emit_json(out, {"ok": True, "active": name})
        return
    emit_text(out, f"active profile: {name}")


@app.command("remove")
def remove_profile(
    ctx: typer.Context,
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete a profile (and its API key)."""
    out = _output(ctx)
    # Cheap confirmation. JSON-mode skips this since it implies scripting.
    if not yes and not out.quiet and not out.json_mode:
        confirm = typer.confirm(f"remove profile '{name}'?", default=False)
        if not confirm:
            emit_text(out, "cancelled.")
            raise typer.Exit(0)
    removed = cfg.remove_profile(name)
    if out.json_mode:
        emit_json(out, {"ok": True, "removed": removed})
        return
    emit_text(out, f"removed: {name}" if removed else f"no such profile: {name}")
