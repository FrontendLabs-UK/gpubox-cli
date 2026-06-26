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
def list_models(
    ctx: typer.Context,
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """List currently hosted models + their tiers."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "GET", "/hosting/models", extra_headers=cfg.workspace_headers(workspace)
        )
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
    name: str = typer.Option(
        ...,
        "--name",
        help="Hosted model name (2-63 chars, [a-z0-9][a-z0-9_-]). Required.",
    ),
    tier: str = typer.Option("cold", "--tier", help="cold|warm|always_hot."),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """Promote a finished run into a hosted endpoint at the given tier."""
    out = _output(ctx)
    _ensure_tier(tier)
    # HostedModelCreate is extra='forbid' and requires training_run_id,
    # hosted_model_name, hosting_tier — send exactly those keys.
    body: dict = {
        "training_run_id": run_id,
        "hosted_model_name": name,
        "hosting_tier": tier,
    }
    with _client(ctx) as client:
        # The training run is looked up under workspace RLS — a run created in
        # workspace A (gpb finetune create sends the header) 404s here without
        # the same workspace header.
        resp = client.request(
            "POST", "/hosting/models", json_body=body, idempotent=True,
            extra_headers=cfg.workspace_headers(workspace),
        )
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
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """Change the tier of a hosted model."""
    out = _output(ctx)
    _ensure_tier(tier)
    with _client(ctx) as client:
        # Tier changes go through POST /hosting/models/{id}/transition; there is
        # no PATCH handler. HostedModelTransition is extra='forbid' with a single
        # required field `hosting_tier`.
        resp = client.request(
            "POST",
            f"/hosting/models/{model_id}/transition",
            json_body={"hosting_tier": tier},
            extra_headers=cfg.workspace_headers(workspace),
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"{model_id} -> tier={tier}")


@app.command("delete")
@exit_on_error
def delete_model(
    ctx: typer.Context,
    model_id: str = typer.Argument(...),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """Delete a hosted model."""
    out = _output(ctx)
    with _client(ctx) as client:
        client.request(
            "DELETE", f"/hosting/models/{model_id}",
            extra_headers=cfg.workspace_headers(workspace),
        )
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
