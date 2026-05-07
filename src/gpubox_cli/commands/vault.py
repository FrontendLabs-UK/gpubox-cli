"""`gpb vault ...` — Conversation Vault (Wave 7.1) and RAG corpora (7.3).

Customers must explicitly enable Vault — we never opt them in by default
(round-table privacy lock from REPL discussion applies here too).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_json, emit_text
from gpubox_cli.version import USER_AGENT

app = typer.Typer(no_args_is_help=True, help="Conversation Vault + RAG corpora.")
conversations = typer.Typer(no_args_is_help=True, help="Vault conversations.")
corpora = typer.Typer(no_args_is_help=True, help="RAG corpora.")

app.add_typer(conversations, name="conversations")
app.add_typer(corpora, name="corpora")


def _output(ctx: typer.Context) -> OutputCtx:
    return (ctx.obj or {}).get("output", OutputCtx())


def _resolve(ctx_obj: dict) -> cfg.ResolvedConfig:
    return cfg.resolve(
        profile_override=ctx_obj.get("profile"),
        api_key_override=ctx_obj.get("api_key"),
        base_url_override=ctx_obj.get("base_url"),
    )


def _client(ctx: typer.Context) -> GPUBoxClient:
    resolved = _resolve(ctx.obj or {})
    return GPUBoxClient(ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url))


@app.command("enable")
@exit_on_error
def enable(ctx: typer.Context) -> None:
    """Opt the current tenant into Vault persistence."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("POST", "/vault/enable", json_body={"enabled": True})
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, "Vault enabled.")


@app.command("disable")
@exit_on_error
def disable(ctx: typer.Context) -> None:
    """Disable Vault persistence (existing conversations are not deleted)."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("POST", "/vault/enable", json_body={"enabled": False})
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, "Vault disabled. Existing conversations remain.")


@app.command("search")
@exit_on_error
def search(
    ctx: typer.Context,
    query: str = typer.Argument(...),
    limit: int = typer.Option(10, "--limit", "-n"),
) -> None:
    """Semantic search across stored conversations."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "POST", "/vault/search", json_body={"query": query, "limit": limit}
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    hits = resp.get("hits", []) if isinstance(resp, dict) else []
    if not hits:
        emit_text(out, "(no matches)")
        return
    for hit in hits:
        emit_text(
            out,
            f"{hit.get('id','?'):<24} score={hit.get('score','?'):<6} "
            f"{hit.get('snippet','')[:80]}",
        )


# ---- conversations ---------------------------------------------------------


@conversations.command("list")
@exit_on_error
def list_convs(ctx: typer.Context, limit: int = typer.Option(20, "--limit", "-n")) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/vault/conversations", params={"limit": limit})
    if out.json_mode:
        emit_json(out, resp)
        return
    for item in (resp.get("items", []) if isinstance(resp, dict) else []):
        emit_text(
            out,
            f"{item.get('id','?'):<24} {item.get('updated_at','?')} "
            f"msgs={item.get('message_count','?')}",
        )


@conversations.command("get")
@exit_on_error
def get_conv(ctx: typer.Context, conv_id: str = typer.Argument(...)) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", f"/vault/conversations/{conv_id}")
    if out.json_mode:
        emit_json(out, resp)
        return
    msgs = resp.get("messages", []) if isinstance(resp, dict) else []
    for msg in msgs:
        emit_text(out, f"[{msg.get('role','?')}] {msg.get('content','')}")


# ---- corpora ---------------------------------------------------------------


@corpora.command("list")
@exit_on_error
def list_corpora(ctx: typer.Context) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/vault/corpora")
    if out.json_mode:
        emit_json(out, resp)
        return
    for item in (resp.get("items", []) if isinstance(resp, dict) else []):
        emit_text(
            out,
            f"{item.get('id','?'):<24} {item.get('name','?'):<32} "
            f"docs={item.get('doc_count','?')}",
        )


@corpora.command("create")
@exit_on_error
def create_corpus(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Human-readable corpus name."),
    from_file: Path | None = typer.Option(
        None,
        "--from-file",
        exists=True,
        readable=True,
        help="Optional initial archive (zip/tar.gz) to upload.",
    ),
) -> None:
    """Create a corpus and optionally upload an initial archive."""
    out = _output(ctx)
    obj = ctx.obj or {}
    resolved = _resolve(obj)
    with GPUBoxClient(
        ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
    ) as client:
        resp = client.request(
            "POST", "/vault/corpora", json_body={"name": name}, idempotent=True
        )

    corpus_id = resp.get("id") if isinstance(resp, dict) else None
    if from_file and corpus_id:
        # Stream the archive multipart — large file friendly.
        url = resolved.base_url.rstrip("/") + f"/vault/corpora/{corpus_id}/upload"
        headers = {
            "Authorization": f"Bearer {resolved.api_key}",
            "User-Agent": USER_AGENT,
        }
        with from_file.open("rb") as fh:
            files = {"file": (from_file.name, fh, "application/octet-stream")}
            try:
                upload = httpx.post(url, headers=headers, files=files, timeout=600.0)
            except httpx.HTTPError as exc:
                from gpubox_cli.client import GPUBoxError

                raise GPUBoxError(f"upload network error: {exc}") from exc

            if upload.status_code >= 400:
                # Use the public helper for typed errors (401/402/etc).
                with GPUBoxClient(
                    ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
                ) as inner:
                    inner.raise_for_response(upload)

    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"corpus: {corpus_id} (name={name})")
