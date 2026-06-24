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

import json
from pathlib import Path

import httpx
import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, GPUBoxError, exit_on_error
from gpubox_cli.output import OutputCtx, emit_json, emit_text
from gpubox_cli.version import USER_AGENT

app = typer.Typer(no_args_is_help=True, help="Conversation Vault + RAG corpora.")
conversations = typer.Typer(no_args_is_help=True, help="Vault conversations.")
corpora = typer.Typer(no_args_is_help=True, help="RAG corpora.")

app.add_typer(conversations, name="conversations")
app.add_typer(corpora, name="corpora")

# The record fields the save endpoint (POST /vault/enrich/save -> EnrichRecordIn)
# accepts. We project preview records down to exactly these so a forward-compatible
# annotation field on the preview response never breaks the save POST.
_SAVE_RECORD_FIELDS = (
    "url",
    "extracted",
    "fetched_at",
    "status",
    "source_host",
    "truncated",
)


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


# ---- web enrichment --------------------------------------------------------


def _load_capture_schema(schema_path: Path | None) -> dict | None:
    """Read + parse an optional JSON-schema file for `--schema`.

    A clean GPUBoxError (not a traceback) on a missing/garbage file so the
    CLI exits with a useful message and the documented exit code.
    """
    if schema_path is None:
        return None
    try:
        raw = schema_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GPUBoxError(f"could not read schema file {schema_path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GPUBoxError(f"schema file {schema_path} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise GPUBoxError(f"schema file {schema_path} must contain a JSON object")
    return parsed


def _read_batch_urls(batch_path: Path) -> list[str]:
    """Read URLs from a --batch file: one per line, blank lines + #comments
    skipped. The gateway caps a single enrich call at 5 URLs, so we surface a
    clean error rather than letting the gateway 422 a too-long list."""
    try:
        lines = batch_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GPUBoxError(f"could not read batch file {batch_path}: {exc}") from exc
    urls = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not urls:
        raise GPUBoxError(f"batch file {batch_path} contained no URLs")
    if len(urls) > 5:
        raise GPUBoxError(
            f"batch file has {len(urls)} URLs; the gateway accepts at most 5 "
            "per enrich call — split the file."
        )
    return urls


@app.command("enrich")
@exit_on_error
def enrich(
    ctx: typer.Context,
    url: str | None = typer.Argument(
        None, help="A single public http(s) URL to enrich. Omit when using --batch."
    ),
    capture: str = typer.Option(
        ...,
        "--capture",
        "-c",
        help="Plain-language description of the data to extract (e.g. 'pricing, contact email').",
    ),
    schema: Path | None = typer.Option(
        None,
        "--schema",
        exists=True,
        readable=True,
        help="Optional JSON-schema file forcing the shape of the extracted output.",
    ),
    batch: Path | None = typer.Option(
        None,
        "--batch",
        exists=True,
        readable=True,
        help="A file of URLs (one per line, 1..5) to enrich in one call instead of a single URL.",
    ),
    collection: str | None = typer.Option(
        None,
        "--collection",
        help="Target vault collection id (used only with --save).",
    ),
    save: bool = typer.Option(
        False,
        "--save/--preview",
        help="Persist accepted records into the Vault. GATED on the DB cutover (returns 404/400 until the flag flips); default is preview-only.",
    ),
    respect_robots: bool = typer.Option(
        True,
        "--respect-robots/--no-respect-robots",
        help="Honour robots.txt (default). Disable only on pages you own.",
    ),
) -> None:
    """Fetch public web page(s) and extract structured data.

    Preview (default) returns the extracted records and writes NOTHING —
    pipe it straight into `jq`. `--save` persists the `ok` records into the
    Vault, but that path is GATED on the DB cutover: until the flag flips the
    gateway returns a typed 404 `save_not_enabled` (or 400 `save_gated` on the
    preview route), which the CLI surfaces cleanly rather than crashing.

    Maps to:
      * preview -> POST /v1/vault/enrich
      * --save  -> POST /v1/vault/enrich (preview) then POST /v1/vault/enrich/save
    """
    out = _output(ctx)

    # --collection only means anything on the save path; silently ignoring it on
    # a preview would discard the user's choice without telling them.
    if collection is not None and not save:
        raise GPUBoxError("--collection only applies with --save (preview writes nothing)")

    # Resolve the URL set: a single positional URL XOR a --batch file.
    if batch is not None and url is not None:
        raise GPUBoxError("pass a single URL or --batch, not both")
    if batch is not None:
        urls = _read_batch_urls(batch)
    elif url is not None:
        urls = [url]
    else:
        raise GPUBoxError("provide a URL argument or --batch <file>")

    schema_obj = _load_capture_schema(schema)

    body: dict = {"urls": urls, "capture": capture, "respect_robots": respect_robots}
    if schema_obj is not None:
        body["schema"] = schema_obj

    with _client(ctx) as client:
        # Always run the preview first. On a 1..5-URL preview the gateway
        # returns one record per URL (partial-success tolerant).
        preview = client.request("POST", "/vault/enrich", json_body=body)

        if not save:
            if out.json_mode:
                emit_json(out, preview)
                return
            _emit_enrich_records(out, preview)
            return

        # --save: project the OK preview records to the save shape and POST
        # them. The save endpoint takes records (not URLs) + an optional
        # collection. A record that did not extract cleanly is NOT saved — but
        # we must NOT swallow it: a non-ok URL never reaches the save response's
        # `skipped` list (it never got POSTed), so we track those preview
        # failures separately and surface them in the output.
        records = preview.get("records", []) if isinstance(preview, dict) else []
        ok_records = [
            {k: r.get(k) for k in _SAVE_RECORD_FIELDS}
            for r in records
            if isinstance(r, dict) and r.get("status") == "ok"
        ]
        preview_skipped = [
            {"url": r.get("url"), "status": r.get("status"), "reason": r.get("reason")}
            for r in records
            if isinstance(r, dict) and r.get("status") != "ok"
        ]
        if not ok_records:
            # Nothing extracted cleanly — emit the preview so the user sees
            # the per-record reasons rather than POSTing an empty save.
            if out.json_mode:
                emit_json(
                    out,
                    {
                        "saved": False,
                        "preview": preview,
                        "preview_skipped": preview_skipped,
                        "skipped_save": "no ok records",
                    },
                )
                return
            emit_text(out, "no records extracted cleanly; nothing to save")
            _emit_enrich_records(out, preview)
            return

        save_body: dict = {"records": ok_records}
        if collection is not None:
            save_body["collection"] = collection
        saved = client.request(
            "POST", "/vault/enrich/save", json_body=save_body, idempotent=True
        )

    if out.json_mode:
        # Merge the preview-stage failures into the save document so a `--json`
        # consumer sees BOTH the saved docs and the URLs that never made it to
        # the save call (otherwise `skipped` undercounts).
        doc = dict(saved) if isinstance(saved, dict) else {"saved": saved}
        if preview_skipped:
            doc["preview_skipped"] = preview_skipped
        emit_json(out, doc)
        return
    if isinstance(saved, dict):
        doc_ids = saved.get("document_ids", []) or []
        save_skipped = saved.get("skipped", []) or []
        emit_text(
            out,
            f"saved={saved.get('saved')} documents={len(doc_ids)} "
            f"save_skipped={len(save_skipped)} preview_skipped={len(preview_skipped)}",
        )
        for did in doc_ids:
            emit_text(out, f"  document {did}")
        for sk in save_skipped:
            if isinstance(sk, dict):
                emit_text(out, f"  save-skipped {sk.get('url','?')}: {sk.get('reason','?')}")
        for sk in preview_skipped:
            emit_text(
                out,
                f"  preview-skipped {sk.get('url','?')}: "
                f"{sk.get('status','?')} {sk.get('reason') or ''}",
            )


def _emit_enrich_records(out: OutputCtx, preview: dict) -> None:
    """Human-readable rendering of a preview response (non --json mode)."""
    records = preview.get("records", []) if isinstance(preview, dict) else []
    if not records:
        emit_text(out, "(no records)")
        return
    for rec in records:
        if not isinstance(rec, dict):
            continue
        host = rec.get("source_host", "?")
        status = rec.get("status", "?")
        flags = []
        if rec.get("pii_detected"):
            flags.append("pii")
        if rec.get("injection_suspected"):
            flags.append("injection?")
        if rec.get("needs_disambiguation"):
            flags.append("ambiguous")
        if rec.get("truncated"):
            flags.append("truncated")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        emit_text(out, f"{host:<28} status={status}{flag_str}")
        reason = rec.get("reason")
        if reason and status != "ok":
            emit_text(out, f"  reason: {reason}")
        extracted = rec.get("extracted") or {}
        if isinstance(extracted, dict):
            for k, v in extracted.items():
                emit_text(out, f"  {k}: {v}")


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
