"""`gpb config get/set` — read/write non-secret settings.

Stored in config.toml (NOT credentials.toml) so this is safe to inspect
casually. Currently supported keys:

* ``active_profile``  — name of the default profile
* ``default_model``   — model id used when --model isn't passed

Other keys are tucked into ``settings.extra`` for forward compatibility;
unknown keys round-trip safely.
"""

from __future__ import annotations

import typer

from gpubox_cli import config as cfg
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Read / write CLI config.")

# Allowlist of "first-class" config keys. Unknown keys still work via
# settings.extra but won't render through `gpb config get` without a key arg.
_KNOWN_KEYS = {"active_profile", "default_model"}


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
