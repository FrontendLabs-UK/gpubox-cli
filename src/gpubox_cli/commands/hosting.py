"""`gpb hosting ...` — promote a finished run to a hosted endpoint and
manage hosting tier (cold | warm | always_hot).

Wave 9 backend. Tiers are domain concepts owned by the gateway; we don't
validate them client-side beyond a small allow-list to catch typos.
"""

from __future__ import annotations

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Hosting tier management.")
_VALID_TIERS = {"cold", "warm", "always_hot"}


def _client(ctx: typer.Context) -> GPUBoxClient:
    obj = ctx.obj or {}
    resolved = cfg.resolve(
        profile_override=obj.get("profile"),
        api_key_override=obj.get("api_key"),
        base_url_override=obj.get("base_url"),
    )
    return GPUBoxClient(ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url))


def _output(ctx: typer.Context) -> OutputCtx:
    return (ctx.obj or {}).get("output", OutputCtx())


@app.command("list")
@exit_on_error
def list_models(ctx: typer.Context) -> None:
    """List currently hosted models + their tiers."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/hosting/models")
    if out.json_mode:
        emit_json(out, resp)
        return
    items = resp.get("items", []) if isinstance(resp, dict) else []
    for item in items:
        emit_text(
            out,
            f"{item.get('id','?'):<32} {item.get('tier','?'):<12} {item.get('status','?')}",
        )
    if not items:
        emit_text(out, "(no hosted models)")


@app.command("promote")
@exit_on_error
def promote(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="Successful training run id."),
    tier: str = typer.Option("cold", "--tier", help="cold|warm|always_hot."),
    name: str | None = typer.Option(None, "--name", help="Override hosted model id."),
) -> None:
    """Promote a finished run into a hosted endpoint at the given tier."""
    out = _output(ctx)
    _ensure_tier(tier)
    body: dict = {"run_id": run_id, "tier": tier}
    if name:
        body["name"] = name
    with _client(ctx) as client:
        resp = client.request("POST", "/hosting/models", json_body=body, idempotent=True)
    if out.json_mode:
        emit_json(out, resp)
        return
    mid = resp.get("id") if isinstance(resp, dict) else None
    emit_text(out, f"hosted: {mid} (tier={tier})")


@app.command("tier")
@exit_on_error
def set_tier(
    ctx: typer.Context,
    model_id: str = typer.Argument(...),
    tier: str = typer.Option(..., "--tier", help="cold|warm|always_hot."),
) -> None:
    """Change the tier of a hosted model."""
    out = _output(ctx)
    _ensure_tier(tier)
    with _client(ctx) as client:
        resp = client.request(
            "PATCH", f"/hosting/models/{model_id}", json_body={"tier": tier}
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"{model_id} -> tier={tier}")


@app.command("delete")
@exit_on_error
def delete_model(ctx: typer.Context, model_id: str = typer.Argument(...)) -> None:
    """Delete a hosted model."""
    out = _output(ctx)
    with _client(ctx) as client:
        client.request("DELETE", f"/hosting/models/{model_id}")
    if out.json_mode:
        emit_json(out, {"ok": True, "id": model_id})
        return
    emit_text(out, f"deleted: {model_id}")


def _ensure_tier(tier: str) -> None:
    if tier not in _VALID_TIERS:
        # Typer will surface this with a non-zero exit because we raise BadParameter.
        raise typer.BadParameter(
            f"tier must be one of {sorted(_VALID_TIERS)}", param_hint="--tier"
        )
