"""`gpb workspace ...` — GPUBox V1.5 W1 Workspaces.

A workspace is an additive isolation container within your tenant — group
conversations, vault docs, assistants, and fine-tunes under a named
workspace ("Personal", "Acme Client", "R&D"). `workspace use` pins an
active workspace in the CLI config; subsequent scoped commands send it as
the X-GPUBox-Workspace header.

Path style: the resolved base_url already ends in /v1, so command paths are
written WITHOUT the /v1 prefix (e.g. "/workspaces", matching the rest of the
CLI).
"""
from __future__ import annotations

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Workspaces — per-tenant isolation containers.")

# CLI config key (stored in Settings.extra) for the pinned active workspace.
_ACTIVE_WORKSPACE_KEY = "active_workspace"


def _output(ctx: typer.Context) -> OutputCtx:
    return (ctx.obj or {}).get("output", OutputCtx())


def _client(ctx: typer.Context) -> GPUBoxClient:
    obj = ctx.obj or {}
    resolved = cfg.resolve(
        profile_override=obj.get("profile"),
        api_key_override=obj.get("api_key"),
        base_url_override=obj.get("base_url"),
    )
    return GPUBoxClient(ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url))


def _active_workspace() -> str | None:
    settings = cfg.load_settings()
    val = settings.extra.get(_ACTIVE_WORKSPACE_KEY)
    return str(val) if val else None


@app.command("create", help="Create a workspace by name.")
@exit_on_error
def create_workspace(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Workspace display name."),
    slug: str | None = typer.Option(None, "--slug", help="Optional URL-friendly handle."),
    description: str | None = typer.Option(None, "--description"),
) -> None:
    """Create a workspace by name."""
    out = _output(ctx)
    body: dict = {"name": name}
    if slug:
        body["slug"] = slug
    if description:
        body["description"] = description
    with _client(ctx) as client:
        resp = client.request("POST", "/workspaces", json_body=body, idempotent=True)
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"created: {resp.get('id', '?')}  {resp.get('name', '')}")


@app.command("list", help="List workspaces in the active tenant.")
@exit_on_error
def list_workspaces(ctx: typer.Context) -> None:
    """List workspaces in the active tenant."""
    out = _output(ctx)
    active = _active_workspace()
    with _client(ctx) as client:
        resp = client.request("GET", "/workspaces")
    if out.json_mode:
        emit_json(out, resp)
        return
    items = resp.get("data", []) if isinstance(resp, dict) else []
    for item in items:
        marker = "*" if active and item.get("id") == active else " "
        default = " (default)" if item.get("is_default") else ""
        emit_text(
            out,
            f"{marker} {item.get('id', '?'):<38} {item.get('name', '?'):<24}{default}",
        )
    if active is None:
        emit_text(out, "\nNo active workspace pinned — `gpb workspace use <id>` to set one.")


@app.command("use", help="Pin an active workspace (sent as X-GPUBox-Workspace on scoped commands).")
@exit_on_error
def use_workspace(
    ctx: typer.Context,
    workspace_id: str = typer.Argument(..., help="Workspace id to activate."),
) -> None:
    """Pin the active workspace in CLI config.

    Validates the id exists + is owned by the calling tenant before pinning,
    so a typo doesn't silently scope every later command to nothing.
    """
    out = _output(ctx)
    with _client(ctx) as client:
        # 404s if the workspace doesn't exist / isn't owned by this tenant.
        resp = client.request("GET", f"/workspaces/{workspace_id}")
    settings = cfg.load_settings()
    settings.extra[_ACTIVE_WORKSPACE_KEY] = workspace_id
    cfg.save_settings(settings)
    if out.json_mode:
        emit_json(out, {"active_workspace": workspace_id, "name": resp.get("name")})
        return
    emit_text(out, f"active workspace set: {workspace_id}  {resp.get('name', '')}")


@app.command(
    "update",
    help="Rename a workspace and/or set its Settings defaults "
         "(model / response language / watch cadence).",
)
@exit_on_error
def update_workspace(
    ctx: typer.Context,
    workspace_id: str = typer.Argument(..., help="Workspace id to update."),
    name: str | None = typer.Option(None, "--name", help="New display name."),
    slug: str | None = typer.Option(None, "--slug", help="New URL-friendly handle."),
    description: str | None = typer.Option(None, "--description"),
    default_model: str | None = typer.Option(
        None, "--default-model",
        help="Default BASE chat model id for the composer "
             "(e.g. qwen2.5-32b-instruct). Fine-tunes default via "
             "`gpb finetune use`, not here.",
    ),
    response_language: str | None = typer.Option(
        None, "--response-language",
        help="Preferred response language code (BCP-47, e.g. en-GB, yo).",
    ),
    watch_cadence: str | None = typer.Option(
        None, "--watch-cadence",
        help="Default Argus watch cadence: manual|hourly|daily|weekly.",
    ),
) -> None:
    """Partially update a workspace.

    Only the flags you pass are changed (omitted fields are left as-is). The
    PATCH is tenant-scoped server-side, so you can only update a workspace your
    own tenant owns. Validation (known model, cadence enum, language shape)
    happens on the gateway; an invalid value returns a 400.
    """
    out = _output(ctx)
    body: dict = {}
    if name is not None:
        body["name"] = name
    if slug is not None:
        body["slug"] = slug
    if description is not None:
        body["description"] = description
    if default_model is not None:
        body["default_model"] = default_model
    if response_language is not None:
        body["response_language"] = response_language
    if watch_cadence is not None:
        body["default_watch_cadence"] = watch_cadence
    if not body:
        emit_error(
            out,
            "nothing to update — pass at least one of --name / --slug / "
            "--description / --default-model / --response-language / "
            "--watch-cadence.",
        )
        raise typer.Exit(2)
    with _client(ctx) as client:
        resp = client.request(
            "PATCH", f"/workspaces/{workspace_id}", json_body=body,
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"updated: {resp.get('id', '?')}  {resp.get('name', '')}")


@app.command("delete", help="Soft-delete a workspace by id (the Default cannot be deleted).")
@exit_on_error
def delete_workspace(
    ctx: typer.Context,
    workspace_id: str = typer.Argument(...),
) -> None:
    """Soft-delete a workspace by id."""
    out = _output(ctx)
    with _client(ctx) as client:
        client.request("DELETE", f"/workspaces/{workspace_id}")
    # If the deleted workspace was the pinned active one, clear the pin.
    settings = cfg.load_settings()
    if settings.extra.get(_ACTIVE_WORKSPACE_KEY) == workspace_id:
        settings.extra.pop(_ACTIVE_WORKSPACE_KEY, None)
        cfg.save_settings(settings)
    if out.json_mode:
        emit_json(out, {"ok": True, "id": workspace_id})
        return
    emit_text(out, f"deleted: {workspace_id}")
