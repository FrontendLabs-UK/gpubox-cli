"""`gpb embed "text"` — return an embedding vector.

Buffered (no streaming applies here). Emits the raw vector when --json,
otherwise prints a one-line summary so a quick `gpb embed "hi"` from a
terminal isn't a wall of floats.
"""

from __future__ import annotations

import sys

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def _build_client(ctx_obj: dict) -> GPUBoxClient:
    resolved = cfg.resolve(
        profile_override=ctx_obj.get("profile"),
        api_key_override=ctx_obj.get("api_key"),
        base_url_override=ctx_obj.get("base_url"),
    )
    return GPUBoxClient(
        ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
    )


@exit_on_error
def run(
    ctx: typer.Context,
    text: str | None = typer.Argument(
        None, help="Text to embed. Pass `-` to read from stdin."
    ),
    model: str = typer.Option(
        DEFAULT_EMBEDDING_MODEL, "--model", "-m", help="Embedding model id."
    ),
) -> None:
    """Generate an embedding vector for a piece of text."""
    ctx_obj = ctx.obj or {}
    out: OutputCtx = ctx_obj.get("output", OutputCtx())

    if text == "-" or text is None:
        if sys.stdin.isatty() and text is None:
            emit_error(out, "missing text. example: gpb embed \"hello world\"")
            raise typer.Exit(2)
        text = sys.stdin.read().strip()
        if not text:
            emit_error(out, "empty input")
            raise typer.Exit(2)

    body = {"model": model, "input": text}
    with _build_client(ctx_obj) as client:
        resp = client.request("POST", "/embeddings", json_body=body)

    if out.json_mode:
        emit_json(out, resp)
        return

    # Human-friendly default — print dim + first few floats so the user
    # knows it worked without scrolling 1000 numbers.
    try:
        vec = resp["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        emit_error(out, "unexpected response shape; pass --json for raw payload")
        raise typer.Exit(1) from exc
    preview = ", ".join(f"{x:.4f}" for x in vec[:6])
    emit_text(out, f"model={model} dim={len(vec)}")
    emit_text(out, f"vector[0:6]=[{preview}, …]")
