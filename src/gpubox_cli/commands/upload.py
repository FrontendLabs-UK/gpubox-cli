"""`gpb upload <file>` — upload a document into your hardened Vault (V1.6).

The hardened DOCUMENT vault uses a two-step signed-URL upload: the gateway mints
a short-lived upload URL + creates the doc row (`scan_status='pending'`), we PUT
the bytes straight to object storage, and the vault workers then advance the row
through the pipeline:

    scan  ->  extract  ->  chunk  ->  embed  ->  ready

Those stages run async, so without watching the status a freshly-uploaded doc
just sits `pending` from the caller's point of view. This command therefore polls
the per-document indexing status and shows real progress until it reaches the
terminal `ready` (grounding-usable in chat) or `failed`. `--no-wait` returns as
soon as the bytes are uploaded (the workers still process it server-side).
"""

from __future__ import annotations

import hashlib
import mimetypes
import re
import time
from pathlib import Path

import httpx
import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, GPUBoxError, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text
from gpubox_cli.version import USER_AGENT

_DEFAULT_COLLECTION = "Uploads"
_TERMINAL_STATES = {"ready", "failed"}
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
# Human labels for the rollup the gateway returns (repo.derive_indexing_state).
_STATE_LABEL = {
    "pending": "queued",
    "queued": "queued",
    "scanning": "scanning for malware",
    "extracting": "extracting text",
    "chunking": "chunking",
    "embedding": "chunking + embedding",
    "indexing": "chunking + embedding",
    "ready": "ready",
    "failed": "failed",
}


def _sha256_and_size(path: Path) -> tuple[str, int]:
    """Stream the file once for the sha256 + byte count the gateway requires
    (it binds the signed upload to this hash, so a corrupted PUT is rejected)."""
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
            total += len(block)
    return h.hexdigest(), total


def _resolve_collection(
    client: GPUBoxClient, collection: str, residency: str, ws_headers: dict[str, str]
) -> str:
    """Return a collection id. A UUID is used as-is; otherwise `collection` is a
    NAME, resolved within the active workspace and created (kind='general') if it
    doesn't exist yet — so a first-time `gpb upload` just works."""
    if _UUID_RE.match(collection):
        return collection
    existing = client.request(
        "GET", "/vault/collections", params={"name": collection},
        extra_headers=ws_headers or None,
    )
    if isinstance(existing, list) and existing:
        return existing[0]["id"]
    created = client.request(
        "POST", "/vault/collections",
        # data_residency is an uppercase enum on the gateway (UK|NG|EU|MULTI).
        json_body={"name": collection, "kind": "general", "data_residency": residency.upper()},
        extra_headers=ws_headers or None,
    )
    return created["id"]


@exit_on_error
def run(
    ctx: typer.Context,
    file_path: Path = typer.Argument(
        ..., exists=True, readable=True, file_okay=True, dir_okay=False,
        resolve_path=True, help="Path to the document to upload.",
    ),
    collection: str = typer.Option(
        _DEFAULT_COLLECTION, "--collection", "-c",
        help="Collection NAME (created if missing) or an existing collection UUID.",
    ),
    data_residency: str = typer.Option(
        "UK", "--residency", help="Data residency for a newly-created collection: UK, NG, EU or MULTI.",
    ),
    workspace: str | None = typer.Option(
        None, "--workspace", "-w", help="Workspace id to upload into (default: active workspace).",
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait",
        help="Poll indexing status until the doc is ready/failed (default), or return after upload.",
    ),
    poll_interval: float = typer.Option(
        2.0, "--poll-interval", help="Seconds between status polls while waiting.",
    ),
    timeout: float = typer.Option(
        300.0, "--timeout", help="Max seconds to wait for indexing before giving up (doc keeps processing server-side).",
    ),
) -> None:
    """Upload a document into your hardened Vault and watch it index."""
    ctx_obj = ctx.obj or {}
    out: OutputCtx = ctx_obj.get("output", OutputCtx())
    resolved = cfg.resolve(
        profile_override=ctx_obj.get("profile"),
        api_key_override=ctx_obj.get("api_key"),
        base_url_override=ctx_obj.get("base_url"),
    )
    if not resolved.api_key:
        emit_error(out, "no API key configured. run `gpb auth login`.")
        raise typer.Exit(4)

    mime, _ = mimetypes.guess_type(file_path.name)
    mime = mime or "application/octet-stream"
    sha256_hex, size_bytes = _sha256_and_size(file_path)
    if size_bytes == 0:
        emit_error(out, f"{file_path.name} is empty (0 bytes).")
        raise typer.Exit(2)

    # Honour the PINNED active workspace (`gpb workspace use`) when --workspace is
    # not given — else a doc lands in Default and vanishes from the user's active
    # workspace search/chat (Codex review HIGH).
    ws_headers = cfg.workspace_headers(workspace) or {}

    with GPUBoxClient(
        ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
    ) as client:
        collection_id = _resolve_collection(client, collection, data_residency, ws_headers)

        # Step 1 — mint the signed upload URL + create the pending doc row.
        try:
            minted = client.request(
                "POST", "/vault/documents",
                json_body={
                    "collection_id": collection_id,
                    "filename": file_path.name,
                    "mime": mime,
                    "size_bytes": size_bytes,
                    "sha256_hex": sha256_hex,
                },
                idempotent=True,
                extra_headers=ws_headers or None,
            )
        except GPUBoxError as exc:
            # The vault dedupes by (tenant, sha256): re-uploading identical bytes
            # is a no-op, not a failure — surface it cleanly.
            msg = str(exc)
            if "409" in msg or "already exists" in msg:
                if out.json_mode:
                    emit_json(out, {"state": "duplicate",
                                    "message": "identical document already in your vault"})
                else:
                    emit_text(out, f"{file_path.name} is already in your vault "
                                   "(identical content) — nothing to upload.")
                return
            raise
        document_id = minted["document_id"]
        upload_url = minted["upload_url"]

        # Step 2 — PUT the bytes to object storage (presigned URL: no auth header).
        if not out.quiet:
            typer.echo(f"uploading {file_path.name} ({size_bytes:,} bytes)…", err=True)
        with file_path.open("rb") as fh:
            try:
                # Stream the handle (not fh.read()) so a large PDF isn't loaded
                # whole into memory; Content-Length is explicit since the signed
                # PUT needs a known length (Codex review MEDIUM).
                put = httpx.put(
                    upload_url, content=fh,
                    headers={
                        "Content-Type": mime,
                        "Content-Length": str(size_bytes),
                        "User-Agent": USER_AGENT,
                    },
                    timeout=httpx.Timeout(300.0, connect=15.0),
                )
            except httpx.HTTPError as exc:
                raise GPUBoxError(f"upload failed (PUT to storage): {exc}") from exc
        if put.status_code >= 400:
            raise GPUBoxError(
                f"storage rejected the upload: HTTP {put.status_code} {put.text[:200]}"
            )

        if not wait:
            result = {"document_id": document_id, "state": "processing",
                      "message": "uploaded; indexing in the background"}
            if out.json_mode:
                emit_json(out, result)
            else:
                emit_text(out, f"uploaded — document {document_id} is indexing (use `--wait` to watch).")
            return

        # Step 3 — poll the indexing rollup until terminal or timeout.
        deadline = time.monotonic() + timeout
        last_label = None
        status: dict = {}
        while True:
            status = client.request(
                "GET", f"/vault/documents/{document_id}/status",
                extra_headers=ws_headers or None,
            )
            state = status.get("indexing_state", "pending")
            label = _STATE_LABEL.get(state, state)
            if label != last_label and not out.json_mode and not out.quiet:
                typer.echo(f"  … {label}", err=True)
                last_label = label
            if state in _TERMINAL_STATES:
                break
            if time.monotonic() >= deadline:
                emit_error(
                    out,
                    f"document {document_id} still '{state}' after {timeout:.0f}s — "
                    "it keeps processing server-side; check the vault in the web UI, "
                    "or re-run with a larger --timeout.",
                )
                raise typer.Exit(5)
            time.sleep(poll_interval)

        state = status.get("indexing_state")
        if state == "failed":
            reason = status.get("failure_reason") or "unknown"
            if out.json_mode:
                emit_json(out, status)
            else:
                emit_error(out, f"document {document_id} FAILED to index: {reason}")
            raise typer.Exit(1)

        if out.json_mode:
            emit_json(out, status)
        else:
            emit_text(out, f"✓ {file_path.name} → document {document_id} is READY (grounding-usable in chat).")
