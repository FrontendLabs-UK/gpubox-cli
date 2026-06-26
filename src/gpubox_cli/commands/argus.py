"""`gpb argus ...` — GPUBox V1.5 W4 Argus, the Standing Research Agent.

Create a standing question Argus keeps answered + updated from your vault,
delivered to an in-app inbox. Read-only (no vault writes, no autonomous sends,
no email), workspace-scoped, grounded + cited. Vault-only sources in V1.

Path style: the resolved base_url already ends in /v1, so command paths are
written WITHOUT the /v1 prefix (e.g. "/argus/agents").
"""
from __future__ import annotations

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Argus — your Standing Research Agent.")

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


def _active_workspace_headers() -> dict:
    settings = cfg.load_settings()
    val = settings.extra.get(_ACTIVE_WORKSPACE_KEY)
    return {"X-GPUBox-Workspace": str(val)} if val else {}


@app.command("create", help="Create a standing research question.")
@exit_on_error
def create_agent(
    ctx: typer.Context,
    question: str = typer.Option(..., "--question", "-q", help="The standing question to keep answered."),
    doc: list[str] = typer.Option(
        ..., "--doc", "-d",
        help="Vault document id(s) the agent may read (repeatable). At least one required.",
    ),
    title: str | None = typer.Option(None, "--title"),
    cadence: str = typer.Option(
        "daily", "--cadence", "-c",
        help="manual | hourly | daily | weekly."),
) -> None:
    """Create a standing research question scoped to specific vault document ids.

    (Tag-scoping is a V1.x feature — the gateway only resolves explicit document
    ids in W4, so the CLI sends an empty tag list.)
    """
    out = _output(ctx)
    body: dict = {
        "question": question,
        "doc_scope_ids": list(doc or []),
        "doc_scope_tags": [],  # tag resolution deferred to V1.x; gateway rejects tags
        "cadence": cadence,
    }
    if title:
        body["title"] = title
    with _client(ctx) as client:
        resp = client.request(
            "POST", "/argus/agents", json_body=body, idempotent=True,
            extra_headers=_active_workspace_headers())
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"created: {resp.get('id', '?')}  {resp.get('question', '')}")


@app.command("list", help="List your standing research questions.")
@exit_on_error
def list_agents(ctx: typer.Context) -> None:
    """List standing questions in the active workspace."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "GET", "/argus/agents", extra_headers=_active_workspace_headers())
    if out.json_mode:
        emit_json(out, resp)
        return
    items = resp.get("data", []) if isinstance(resp, dict) else []
    if not items:
        emit_text(out, "No standing agents. `gpb argus create -q \"...\" -d <doc-id>` to add one.")
        return
    for item in items:
        emit_text(
            out,
            f"{item.get('id', '?'):<38} [{item.get('cadence', '?'):<6}] "
            f"{item.get('status', '?'):<7} {item.get('question', '?')}")


@app.command("get", help="Get one standing research question.")
@exit_on_error
def get_agent(ctx: typer.Context, agent_id: str = typer.Argument(...)) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "GET", f"/argus/agents/{agent_id}",
            extra_headers=_active_workspace_headers())
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"{resp.get('id')}  [{resp.get('cadence')}] {resp.get('status')}")
    emit_text(out, f"  Q: {resp.get('question')}")
    emit_text(out, f"  scope docs: {resp.get('doc_scope_ids')}")


@app.command("run", help="Run a check now (on-demand) for a standing agent.")
@exit_on_error
def run_agent(ctx: typer.Context, agent_id: str = typer.Argument(...)) -> None:
    """Trigger an on-demand check ("Run a check") for a standing agent.

    Queues a run that Argus's scheduler/worker picks up and executes through the
    same bounded, read-only loop the scheduled cadence uses — the inform lands in
    the inbox (`gpb argus inbox`). At most ONE check runs per agent at a time: if
    one is already queued or running, the gateway returns a conflict (run already
    in flight) rather than starting a second.
    """
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "POST", f"/argus/agents/{agent_id}/run", idempotent=True,
            extra_headers=_active_workspace_headers())
    if out.json_mode:
        emit_json(out, resp)
        return
    rid = (resp.get("run_id") or resp.get("id", "?")) if isinstance(resp, dict) else "?"
    status = resp.get("status", "pending") if isinstance(resp, dict) else "pending"
    emit_text(out, f"queued: run {rid} [{status}] — check `gpb argus inbox` shortly.")


@app.command("delete", help="Retire a standing research question.")
@exit_on_error
def delete_agent(ctx: typer.Context, agent_id: str = typer.Argument(...)) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "DELETE", f"/argus/agents/{agent_id}",
            extra_headers=_active_workspace_headers())
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"retired: {agent_id}")


@app.command("inbox", help="Read the Argus inbox (delivered informs).")
@exit_on_error
def read_inbox(
    ctx: typer.Context,
    unread: bool = typer.Option(False, "--unread", help="Only unread informs."),
    limit: int = typer.Option(50, "--limit", "-n"),
) -> None:
    """Read the in-app inbox — grounded, cited informs Argus delivered."""
    out = _output(ctx)
    params = {"unread": "true" if unread else "false", "limit": str(limit)}
    with _client(ctx) as client:
        resp = client.request(
            "GET", "/argus/inbox", params=params,
            extra_headers=_active_workspace_headers())
    if out.json_mode:
        emit_json(out, resp)
        return
    items = resp.get("data", []) if isinstance(resp, dict) else []
    if not items:
        emit_text(out, "Inbox empty.")
        return
    for it in items:
        flag = " " if it.get("read") else "*"
        conf = it.get("confidence") or "?"
        emit_text(out, f"{flag} [{conf:<6}] {it.get('id', '?'):<38} {it.get('title', '')}")
        emit_text(out, f"    {it.get('body', '')[:200]}")


@app.command("read", help="Mark an inform as read.")
@exit_on_error
def mark_read(ctx: typer.Context, item_id: str = typer.Argument(...)) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "POST", f"/argus/inbox/{item_id}/read", idempotent=True,
            extra_headers=_active_workspace_headers())
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"read: {item_id}")
