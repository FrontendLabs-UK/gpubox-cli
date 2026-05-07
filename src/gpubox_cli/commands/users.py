"""`gpb users ...` and `gpb users oidc ...` — Wave 7.5 SSO + ACL admin.

Backend contracts (verified against gpubox-gateway prod):

* POST   /v1/tenants/{tenant_id}/users       — invite (email, name?, role)
* GET    /v1/tenants/{tenant_id}/users       — list (returns ``users`` key, not ``items``)
* PATCH  /v1/tenants/{tenant_id}/users/{user_id}
* POST   /v1/oidc/clients                    — redirect_uris[] (list, not singular)
* GET    /v1/oidc/clients

These all require the tenant_id in the path. We resolve it from
``--tenant`` on the command, ``GPUBOX_TENANT_ID`` env, or the user's
``settings.extra.tenant_id`` (set via ``gpb config set tenant_id <uuid>``).
Without a tenant_id we error with a clear message rather than silently 404.
"""

from __future__ import annotations

import os

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text

ENV_TENANT_ID = "GPUBOX_TENANT_ID"

app = typer.Typer(no_args_is_help=True, help="Users, invites, OIDC clients.")
oidc = typer.Typer(no_args_is_help=True, help="OIDC client management.")
app.add_typer(oidc, name="oidc")


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


def _resolve_tenant(ctx: typer.Context, override: str | None) -> str:
    """Pick a tenant_id in priority order: --tenant > env > profile.extra.

    Profiles can opt into a default tenant via ``gpb config set tenant_id <uuid>``;
    we surface the value through ``settings.extra``. Failing all three we
    raise so the user sees a clear error rather than a 404 with a path
    they probably can't decode.
    """
    if override:
        return override
    env = os.environ.get(ENV_TENANT_ID)
    if env:
        return env
    settings = cfg.load_settings()
    extra = settings.extra
    if isinstance(extra, dict) and isinstance(extra.get("tenant_id"), str):
        return extra["tenant_id"]
    out = _output(ctx)
    emit_error(
        out,
        "tenant_id required for this command. set one of: "
        "--tenant <uuid>, GPUBOX_TENANT_ID env, or `gpb config set tenant_id <uuid>`",
    )
    raise typer.Exit(2)


@app.command("invite")
@exit_on_error
def invite_user(
    ctx: typer.Context,
    email: str = typer.Argument(...),
    role: str = typer.Option("editor", "--role", help="viewer|editor|admin"),
    name: str | None = typer.Option(None, "--name"),
    tenant: str | None = typer.Option(None, "--tenant", help="Tenant UUID."),
) -> None:
    """Invite a teammate by email."""
    out = _output(ctx)
    if role not in {"viewer", "editor", "admin"}:
        emit_error(out, "role must be viewer, editor, or admin")
        raise typer.Exit(2)
    tenant_id = _resolve_tenant(ctx, tenant)
    body: dict = {"email": email, "role": role}
    if name:
        body["name"] = name
    with _client(ctx) as client:
        resp = client.request("POST", f"/tenants/{tenant_id}/users", json_body=body)
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"invited {email} as {role}")


@app.command("list")
@exit_on_error
def list_users(
    ctx: typer.Context,
    tenant: str | None = typer.Option(None, "--tenant", help="Tenant UUID."),
) -> None:
    out = _output(ctx)
    tenant_id = _resolve_tenant(ctx, tenant)
    with _client(ctx) as client:
        resp = client.request("GET", f"/tenants/{tenant_id}/users")
    if out.json_mode:
        emit_json(out, resp)
        return
    # Server returns the rows under ``users`` (not ``items``).
    # Each row uses ``user_id`` (not ``id``) per gateway contract.
    users = (resp.get("users") if isinstance(resp, dict) else None) or []
    for u in users:
        emit_text(
            out,
            f"{u.get('user_id','?'):<36} {u.get('email','?'):<32} "
            f"{u.get('role','?'):<8} {u.get('status','?')}",
        )


# ---- OIDC clients ----------------------------------------------------------


@oidc.command("create")
@exit_on_error
def create_client(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Human-readable client label."),
    redirect_uri: list[str] = typer.Option(
        ..., "--redirect-uri", help="Allowed redirect URI (repeat for multiple)."
    ),
    client_type: str = typer.Option(
        "confidential", "--client-type", help="confidential|public"
    ),
) -> None:
    """Register an OIDC client.

    The gateway requires ``redirect_uris`` as a LIST (Codex caught us
    sending a single string). Repeat ``--redirect-uri`` for multiple.
    """
    out = _output(ctx)
    if client_type not in {"confidential", "public"}:
        emit_error(out, "client-type must be confidential or public")
        raise typer.Exit(2)
    body = {
        "name": name,
        "client_type": client_type,
        "redirect_uris": redirect_uri,
    }
    with _client(ctx) as client:
        resp = client.request("POST", "/oidc/clients", json_body=body)
    if out.json_mode:
        emit_json(out, resp)
        return
    if isinstance(resp, dict):
        emit_text(out, f"client_id: {resp.get('client_id','?')}")
        # client_secret is returned ONCE — point users at --json to capture
        # but never echo it inline (default human output).
        if "client_secret" in resp:
            emit_text(
                out,
                "(client_secret returned in this response — re-run with --json to capture it; "
                "the server will NOT show it again)",
            )


@oidc.command("list")
@exit_on_error
def list_clients(ctx: typer.Context) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/oidc/clients")
    if out.json_mode:
        emit_json(out, resp)
        return
    # Server returns ``clients`` not ``items`` — Codex flagged this.
    clients = (resp.get("clients") if isinstance(resp, dict) else None) or []
    for item in clients:
        emit_text(out, f"{item.get('client_id','?'):<32} {item.get('name','?')}")
