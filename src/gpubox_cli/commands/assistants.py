"""`gpb assistants ...` — Wave 7.4 custom assistants.

Customers create reusable assistant personas with system instructions
loaded from a file (so they can version-control their prompts).
"""

from __future__ import annotations

from pathlib import Path

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Custom assistants.")


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


def _read_instructions(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


@app.command("list")
@exit_on_error
def list_assistants(ctx: typer.Context) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/assistants")
    if out.json_mode:
        emit_json(out, resp)
        return
    for item in (resp.get("items", []) if isinstance(resp, dict) else []):
        emit_text(out, f"{item.get('id','?'):<24} {item.get('name','?'):<32} {item.get('model','')}")


@app.command("create")
@exit_on_error
def create_assistant(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name"),
    instructions: Path = typer.Option(
        ..., "--instructions", exists=True, readable=True, help="Path to prompt file."
    ),
    model: str | None = typer.Option(None, "--model"),
) -> None:
    out = _output(ctx)
    body: dict = {"name": name, "instructions": _read_instructions(instructions)}
    if model:
        body["model"] = model
    with _client(ctx) as client:
        resp = client.request("POST", "/assistants", json_body=body, idempotent=True)
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"created: {resp.get('id','?')}")


@app.command("update")
@exit_on_error
def update_assistant(
    ctx: typer.Context,
    assistant_id: str = typer.Argument(...),
    instructions: Path | None = typer.Option(
        None, "--instructions", exists=True, readable=True
    ),
    name: str | None = typer.Option(None, "--name"),
    model: str | None = typer.Option(None, "--model"),
) -> None:
    out = _output(ctx)
    body: dict = {}
    if instructions:
        body["instructions"] = _read_instructions(instructions)
    if name:
        body["name"] = name
    if model:
        body["model"] = model
    if not body:
        emit_error(out, "nothing to update; pass --instructions, --name, or --model")
        raise typer.Exit(2)
    with _client(ctx) as client:
        resp = client.request("PATCH", f"/assistants/{assistant_id}", json_body=body)
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"updated: {assistant_id}")


@app.command("run")
@exit_on_error
def run_assistant(
    ctx: typer.Context,
    assistant_id: str = typer.Argument(...),
    prompt: str = typer.Argument(...),
) -> None:
    """One-shot run of an assistant with a fresh prompt."""
    out = _output(ctx)
    body = {"input": prompt}
    with _client(ctx) as client:
        resp = client.request("POST", f"/assistants/{assistant_id}/runs", json_body=body)
    if out.json_mode:
        emit_json(out, resp)
        return
    text = (
        resp.get("output", {}).get("content")
        if isinstance(resp, dict) and isinstance(resp.get("output"), dict)
        else None
    ) or resp.get("output") if isinstance(resp, dict) else str(resp)
    emit_text(out, str(text))


@app.command("delete")
@exit_on_error
def delete_assistant(ctx: typer.Context, assistant_id: str = typer.Argument(...)) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        client.request("DELETE", f"/assistants/{assistant_id}")
    if out.json_mode:
        emit_json(out, {"ok": True, "id": assistant_id})
        return
    emit_text(out, f"deleted: {assistant_id}")
