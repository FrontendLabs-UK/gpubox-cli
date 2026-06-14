"""`gpb vault ...` — Conversation Vault (Wave 7.1) and RAG corpora (7.3).

GPUB-403: the whole command group 404'd because every path carried a bogus
`/vault` infix (e.g. `/vault/conversations`) AND the CLI base_url already ends
in `/v1`. The real gateway routes have NO `/vault` segment — they are
`/v1/conversations`, `/v1/conversations/search`, `/v1/corpora`, etc. Because
the resolved base_url is `https://api.gpubox.ai/v1`, the command paths here are
written WITHOUT the `/v1` prefix (httpx joins them onto the base_url path) and
WITHOUT the `/vault` infix. See app/vault.py + app/vault_rag/router.py in the
gpubox-gateway repo for the canonical route table + request/response shapes.

Vault enable/disable: there is NO public route to toggle `tenant.vault_enabled`.
The gateway documents this as an operator-only action (manual SQL / admin UI is
a future wave; `_require_vault_enabled` returns 403 "Contact support@gpubox.ai
to enable" when off). So `gpb vault enable/disable` cannot POST anywhere without
404ing — they now print a clear "not available via CLI" message instead.
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


# Operator-only enablement message — shared by enable/disable so the copy
# stays consistent. Kept here, not inlined, per the "single source of truth"
# convention in the codebase.
_ENABLE_OPERATOR_ONLY = (
    "Vault enablement is operator-only — there is no public API endpoint to "
    "toggle it. Email support@gpubox.ai to have Vault enabled (or disabled) "
    "for your tenant. Once enabled, `gpb vault conversations`, "
    "`gpb vault search`, and `gpb vault corpora` work against your account."
)


@app.command("enable")
@exit_on_error
def enable(ctx: typer.Context) -> None:
    """Vault enablement is operator-only (no public endpoint).

    There is no gateway route to flip `tenant.vault_enabled`; the previous
    POST to `/vault/enable` always 404'd. Email support@gpubox.ai instead.
    """
    out = _output(ctx)
    if out.json_mode:
        emit_json(
            out,
            {
                "enabled": None,
                "available_via_cli": False,
                "message": _ENABLE_OPERATOR_ONLY,
                "contact": "support@gpubox.ai",
            },
        )
        return
    emit_text(out, _ENABLE_OPERATOR_ONLY)


@app.command("disable")
@exit_on_error
def disable(ctx: typer.Context) -> None:
    """Vault disablement is operator-only (no public endpoint).

    Mirrors `enable` — there is no gateway route to flip the tenant flag.
    Email support@gpubox.ai to disable Vault for your tenant.
    """
    out = _output(ctx)
    if out.json_mode:
        emit_json(
            out,
            {
                "enabled": None,
                "available_via_cli": False,
                "message": _ENABLE_OPERATOR_ONLY,
                "contact": "support@gpubox.ai",
            },
        )
        return
    emit_text(out, _ENABLE_OPERATOR_ONLY)


@app.command("search")
@exit_on_error
def search(
    ctx: typer.Context,
    query: str = typer.Argument(...),
    mode: str = typer.Option(
        "fts",
        "--mode",
        "-m",
        help="'fts' (ranked keyword search, default) or 'substring' (exact match).",
    ),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """Keyword search across stored conversations (Postgres FTS).

    This is keyword full-text search, NOT semantic/vector search — the
    gateway route is POST /conversations/search backed by tsvector ('fts')
    or pg_trgm ('substring'). Semantic recall over conversations is a
    separate `/rag/retrieve` surface, not exposed by this command.
    """
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "POST",
            "/conversations/search",
            json_body={"query": query, "mode": mode, "limit": limit},
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    # Gateway returns {"object": "list", "data": [SearchHit, ...]}.
    hits = resp.get("data", []) if isinstance(resp, dict) else []
    if not hits:
        emit_text(out, "(no matches)")
        return
    for hit in hits:
        snippet = (hit.get("snippet", "") or "").replace("\n", " ")[:80]
        emit_text(
            out,
            f"{hit.get('conversation_id','?'):<38} "
            f"rank={hit.get('rank','?'):<6} {snippet}",
        )


# ---- conversations ---------------------------------------------------------


@conversations.command("list")
@exit_on_error
def list_convs(ctx: typer.Context, limit: int = typer.Option(20, "--limit", "-n")) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/conversations", params={"limit": limit})
    if out.json_mode:
        emit_json(out, resp)
        return
    # Gateway returns {"object": "list", "data": [ConversationOut, ...], ...}.
    for item in (resp.get("data", []) if isinstance(resp, dict) else []):
        emit_text(
            out,
            f"{item.get('id','?'):<38} {item.get('last_message_at') or item.get('updated_at','?')} "
            f"msgs={item.get('message_count','?')} {item.get('name') or ''}",
        )


@conversations.command("get")
@exit_on_error
def get_conv(ctx: typer.Context, conv_id: str = typer.Argument(...)) -> None:
    """Fetch a conversation's metadata + its message history."""
    out = _output(ctx)
    with _client(ctx) as client:
        meta = client.request("GET", f"/conversations/{conv_id}")
        msgs_resp = client.request(
            "GET", f"/conversations/{conv_id}/messages", params={"order": "asc"}
        )
    if out.json_mode:
        # Combine metadata + messages into one JSON document so stdout stays
        # a single parseable object (output.py lock #6).
        emit_json(out, {"conversation": meta, "messages": msgs_resp})
        return
    if isinstance(meta, dict):
        emit_text(
            out,
            f"{meta.get('id','?')} {meta.get('name') or ''} "
            f"msgs={meta.get('message_count','?')}",
        )
    # Messages live under {"object": "list", "data": [MessageOut, ...]}.
    msgs = msgs_resp.get("data", []) if isinstance(msgs_resp, dict) else []
    for msg in msgs:
        content = msg.get("content")
        if not isinstance(content, str):
            # Multimodal content is a list-of-parts; render a compact marker.
            content = "[non-text content]"
        emit_text(out, f"[{msg.get('role','?')}] {content}")


@conversations.command("delete")
@exit_on_error
def delete_conv(ctx: typer.Context, conv_id: str = typer.Argument(...)) -> None:
    """Soft-delete a conversation (DELETE /conversations/{id})."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("DELETE", f"/conversations/{conv_id}")
    if out.json_mode:
        # 204 No Content → resp is None; emit a stable success doc.
        emit_json(out, {"deleted": True, "id": conv_id} if resp is None else resp)
        return
    emit_text(out, f"deleted conversation {conv_id}")


# ---- corpora ---------------------------------------------------------------


@corpora.command("list")
@exit_on_error
def list_corpora(ctx: typer.Context, limit: int = typer.Option(50, "--limit", "-n")) -> None:
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/corpora", params={"limit": limit})
    if out.json_mode:
        emit_json(out, resp)
        return
    # Gateway returns {"object": "list", "data": [CorpusResponse, ...]}.
    for item in (resp.get("data", []) if isinstance(resp, dict) else []):
        emit_text(
            out,
            f"{item.get('id','?'):<38} {item.get('name','?'):<32} "
            f"chunks={item.get('chunk_count','?')}",
        )


@corpora.command("get")
@exit_on_error
def get_corpus(ctx: typer.Context, corpus_id: str = typer.Argument(...)) -> None:
    """Fetch a corpus's metadata (GET /corpora/{id})."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", f"/corpora/{corpus_id}")
    if out.json_mode:
        emit_json(out, resp)
        return
    if isinstance(resp, dict):
        emit_text(
            out,
            f"{resp.get('id','?')} {resp.get('name','?')} "
            f"chunks={resp.get('chunk_count','?')} "
            f"embedded={resp.get('embedded_chunk_count','?')} "
            f"bytes={resp.get('total_bytes','?')}",
        )


@corpora.command("delete")
@exit_on_error
def delete_corpus(ctx: typer.Context, corpus_id: str = typer.Argument(...)) -> None:
    """Soft-delete a corpus (DELETE /corpora/{id})."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("DELETE", f"/corpora/{corpus_id}")
    if out.json_mode:
        emit_json(out, {"deleted": True, "id": corpus_id} if resp is None else resp)
        return
    emit_text(out, f"deleted corpus {corpus_id}")


@corpora.command("create")
@exit_on_error
def create_corpus(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Human-readable corpus name."),
    source_type: str = typer.Option(
        "manual",
        "--source-type",
        help="One of: manual, markdown, url. (For PDFs use --from-file.)",
    ),
    content: str | None = typer.Option(
        None,
        "--content",
        help="Inline text (manual/markdown) or an http(s) URL (url source_type).",
    ),
    from_file: Path | None = typer.Option(
        None,
        "--from-file",
        exists=True,
        readable=True,
        help="PDF file to upload (uses the multipart /corpora/upload route).",
    ),
) -> None:
    """Create a corpus.

    Two paths, matching the two gateway routes:

    * `--from-file <pdf>` → multipart POST /corpora/upload (PDF extract).
    * otherwise           → JSON POST /corpora with
      {name, source_type, content} (manual/markdown text or a URL).

    The gateway does NOT accept a bare {name} create — `source_type` and
    (for non-PDF) `content` are required. This is why the old `{name}`-only
    POST would have failed validation even once the path was fixed.
    """
    out = _output(ctx)
    obj = ctx.obj or {}
    resolved = _resolve(obj)

    if from_file is not None:
        # Multipart upload path → POST /v1/corpora/upload. The gateway reads
        # `file`, `name`, `visibility`, `chunk_strategy` as form fields.
        url = resolved.base_url.rstrip("/") + "/corpora/upload"
        headers = {
            "Authorization": f"Bearer {resolved.api_key}",
            "User-Agent": USER_AGENT,
        }
        with from_file.open("rb") as fh:
            files = {"file": (from_file.name, fh, "application/pdf")}
            data = {"name": name}
            try:
                upload = httpx.post(
                    url, headers=headers, files=files, data=data, timeout=600.0
                )
            except httpx.HTTPError as exc:
                from gpubox_cli.client import GPUBoxError

                raise GPUBoxError(f"upload network error: {exc}") from exc

        with GPUBoxClient(
            ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
        ) as inner:
            if upload.status_code >= 400:
                inner.raise_for_response(upload)
        resp = upload.json() if upload.content else {}
        corpus_id = resp.get("id") if isinstance(resp, dict) else None
        if out.json_mode:
            emit_json(out, resp)
            return
        emit_text(out, f"corpus: {corpus_id} (name={name}, from-file={from_file.name})")
        return

    # JSON create path → POST /v1/corpora.
    body: dict = {"name": name, "source_type": source_type}
    if content is not None:
        body["content"] = content
    with GPUBoxClient(
        ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
    ) as client:
        resp = client.request("POST", "/corpora", json_body=body, idempotent=True)

    corpus_id = resp.get("id") if isinstance(resp, dict) else None
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"corpus: {corpus_id} (name={name})")
