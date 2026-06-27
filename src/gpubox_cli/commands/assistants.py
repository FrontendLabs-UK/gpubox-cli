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


@app.command("list", help="List custom assistants in the active workspace.")
@exit_on_error
def list_assistants(
    ctx: typer.Context,
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """List custom assistants in the active workspace."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "GET", "/assistants", extra_headers=cfg.workspace_headers(workspace)
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    for item in (resp.get("items", []) if isinstance(resp, dict) else []):
        emit_text(out, f"{item.get('id','?'):<24} {item.get('name','?'):<32} {item.get('model','')}")


@app.command(
    "create",
    help="Create a custom assistant from a slug, name, and a prompt file.",
)
@exit_on_error
def create_assistant(
    ctx: typer.Context,
    slug: str = typer.Option(
        ...,
        "--slug",
        help=(
            "URL-safe identifier (3-64 chars, [a-z0-9_-], starting + ending "
            "alphanumeric; no '@' or ':'). Required by the gateway."
        ),
    ),
    name: str = typer.Option(..., "--name"),
    instructions: Path = typer.Option(
        ..., "--instructions", exists=True, readable=True, help="Path to prompt file."
    ),
    model: str | None = typer.Option(None, "--model"),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """Create a custom assistant from a slug, name, and a prompt file."""
    out = _output(ctx)
    # CreateAssistantRequest is extra='forbid' and requires both `slug` and
    # `name`. We send exactly the declared fields — slug, name, instructions,
    # and (optionally) model — and nothing else.
    body: dict = {
        "slug": slug,
        "name": name,
        "instructions": _read_instructions(instructions),
    }
    if model:
        body["model"] = model
    with _client(ctx) as client:
        resp = client.request(
            "POST", "/assistants", json_body=body, idempotent=True,
            extra_headers=cfg.workspace_headers(workspace),
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"created: {resp.get('id','?')}")


@app.command(
    "update",
    help="Update an assistant's name, model, or instructions file.",
)
@exit_on_error
def update_assistant(
    ctx: typer.Context,
    assistant_id: str = typer.Argument(...),
    instructions: Path | None = typer.Option(
        None, "--instructions", exists=True, readable=True
    ),
    name: str | None = typer.Option(None, "--name"),
    model: str | None = typer.Option(None, "--model"),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """Update an assistant's name, model, or instructions file."""
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
        # The gateway registers update as POST /v1/assistants/{id} (it creates a
        # new version); there is no PATCH handler, so PATCH 405s. The body shape
        # (name/instructions/model) is already valid for UpdateAssistantRequest.
        resp = client.request(
            "POST", f"/assistants/{assistant_id}", json_body=body,
            extra_headers=cfg.workspace_headers(workspace),
        )
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
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """One-shot run of an assistant with a fresh prompt.

    There is no dedicated `/assistants/{id}/runs` endpoint. An assistant runs
    through the OpenAI-compatible chat surface using the `asst_<uuid>` model
    alias (the gateway resolves the alias to the assistant's pinned version,
    instructions, tools, and corpora). We send a single user turn; pin a
    specific version with an `asst_<id>@vN` argument.
    """
    out = _output(ctx)
    model = assistant_id if assistant_id.startswith("asst_") else f"asst_{assistant_id}"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    with _client(ctx) as client:
        resp = client.request(
            "POST", "/chat/completions", json_body=body,
            extra_headers=cfg.workspace_headers(workspace),
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    text = None
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict):
                text = msg.get("content")
    emit_text(out, str(text if text is not None else resp))


@app.command("delete", help="Delete a custom assistant by id.")
@exit_on_error
def delete_assistant(
    ctx: typer.Context,
    assistant_id: str = typer.Argument(...),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """Delete a custom assistant by id."""
    out = _output(ctx)
    with _client(ctx) as client:
        client.request(
            "DELETE", f"/assistants/{assistant_id}",
            extra_headers=cfg.workspace_headers(workspace),
        )
    if out.json_mode:
        emit_json(out, {"ok": True, "id": assistant_id})
        return
    emit_text(out, f"deleted: {assistant_id}")
