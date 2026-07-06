"""Tests for `gpb upload <file>` — hardened-vault two-step upload + progress poll.

Covers: the happy ready path, the sha256 dedup (409), --no-wait, an indexing
failure, the timeout, and BOTH workspace-header paths (explicit --workspace AND
the pinned active workspace — the Codex-HIGH regression where a pinned workspace
was ignored and the doc silently landed in Default).
"""

from __future__ import annotations

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app

BASE = cfg.DEFAULT_API_URL
UPLOAD_URL = "https://storage.example/r2/put-object?sig=abc"
COLL = "11111111-2222-3333-4444-555555555555"  # a UUID -> skips collection resolve


@pytest.fixture(autouse=True)
def authed(fake_api_key: str) -> None:
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key, base_url=BASE))


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello hardened vault upload\n")
    return p


def _mint() -> httpx.Response:
    return httpx.Response(
        201, json={"document_id": "doc-1", "upload_url": UPLOAD_URL, "expires_in_seconds": 600}
    )


def _status(state: str, *, indexed: bool = False, **extra) -> httpx.Response:
    body = {"id": "doc-1", "scan_status": "clean", "extraction_status": "ok",
            "chunk_status": "ok" if indexed else "pending",
            "indexing_state": state, "indexed": indexed}
    body.update(extra)
    return httpx.Response(200, json=body)


@respx.mock
def test_upload_ready(runner: CliRunner, doc) -> None:
    respx.post(f"{BASE}/vault/documents").mock(return_value=_mint())
    put = respx.put(UPLOAD_URL).mock(return_value=httpx.Response(200))
    respx.get(f"{BASE}/vault/documents/doc-1/status").mock(
        side_effect=[_status("embedding"), _status("ready", indexed=True)]
    )
    r = runner.invoke(app, ["upload", str(doc), "-c", COLL, "--poll-interval", "0.01"])
    assert r.exit_code == 0, r.stderr
    assert "READY" in r.stdout and "doc-1" in r.stdout
    assert put.called
    # mint carried the exact hardened-vault contract fields + an idempotency key.
    sent = put.calls.last.request  # (put body is the file bytes)
    mint_req = respx.calls[0].request
    import json as _json
    body = _json.loads(respx.calls[0].request.read())
    assert set(body) == {"collection_id", "filename", "mime", "size_bytes", "sha256_hex"}
    assert body["collection_id"] == COLL
    assert "Idempotency-Key" in mint_req.headers


@respx.mock
def test_upload_duplicate_sha_is_graceful(runner: CliRunner, doc) -> None:
    respx.post(f"{BASE}/vault/documents").mock(
        return_value=httpx.Response(409, json={"error": {
            "message": "document with this sha256 already exists for this tenant",
            "code": "conflict"}})
    )
    r = runner.invoke(app, ["upload", str(doc), "-c", COLL])
    assert r.exit_code == 0
    assert "already in your vault" in r.stdout


@respx.mock
def test_upload_no_wait_skips_polling(runner: CliRunner, doc) -> None:
    respx.post(f"{BASE}/vault/documents").mock(return_value=_mint())
    put = respx.put(UPLOAD_URL).mock(return_value=httpx.Response(200))
    status = respx.get(f"{BASE}/vault/documents/doc-1/status").mock(
        return_value=httpx.Response(200, json={})
    )
    r = runner.invoke(app, ["upload", str(doc), "-c", COLL, "--no-wait"])
    assert r.exit_code == 0
    assert put.called and not status.called


@respx.mock
def test_upload_failed_surfaces_reason(runner: CliRunner, doc) -> None:
    respx.post(f"{BASE}/vault/documents").mock(return_value=_mint())
    respx.put(UPLOAD_URL).mock(return_value=httpx.Response(200))
    respx.get(f"{BASE}/vault/documents/doc-1/status").mock(
        return_value=_status("failed", extraction_status="errored",
                             failure_reason="could not extract text")
    )
    r = runner.invoke(app, ["upload", str(doc), "-c", COLL, "--poll-interval", "0.01"])
    assert r.exit_code == 1
    assert "could not extract text" in (r.stdout + r.stderr)


@respx.mock
def test_upload_timeout_exits_5(runner: CliRunner, doc) -> None:
    respx.post(f"{BASE}/vault/documents").mock(return_value=_mint())
    respx.put(UPLOAD_URL).mock(return_value=httpx.Response(200))
    respx.get(f"{BASE}/vault/documents/doc-1/status").mock(return_value=_status("embedding"))
    r = runner.invoke(
        app, ["upload", str(doc), "-c", COLL, "--poll-interval", "0.01", "--timeout", "0.02"]
    )
    assert r.exit_code == 5


@respx.mock
def test_upload_explicit_workspace_header(runner: CliRunner, doc) -> None:
    mint = respx.post(f"{BASE}/vault/documents").mock(return_value=_mint())
    respx.put(UPLOAD_URL).mock(return_value=httpx.Response(200))
    r = runner.invoke(app, ["upload", str(doc), "-c", COLL, "--no-wait", "-w", "ws-A"])
    assert r.exit_code == 0, r.stderr
    assert mint.calls.last.request.headers.get("X-GPUBox-Workspace") == "ws-A"


@respx.mock
def test_upload_pinned_workspace_header_rides(runner: CliRunner, doc) -> None:
    # Codex HIGH regression: a PINNED active workspace (no --workspace flag) must
    # still stamp the header, or the doc lands in Default and vanishes from the
    # user's active-workspace search/chat.
    s = cfg.load_settings()
    s.extra["active_workspace"] = "ws-pinned"
    cfg.save_settings(s)
    mint = respx.post(f"{BASE}/vault/documents").mock(return_value=_mint())
    respx.put(UPLOAD_URL).mock(return_value=httpx.Response(200))
    r = runner.invoke(app, ["upload", str(doc), "-c", COLL, "--no-wait"])
    assert r.exit_code == 0, r.stderr
    assert mint.calls.last.request.headers.get("X-GPUBox-Workspace") == "ws-pinned"


@respx.mock
def test_upload_resolves_collection_by_name(runner: CliRunner, doc) -> None:
    # No UUID -> resolve by name: GET (empty) then POST create with UPPERCASE residency.
    respx.get(f"{BASE}/vault/collections").mock(return_value=httpx.Response(200, json=[]))
    create = respx.post(f"{BASE}/vault/collections").mock(
        return_value=httpx.Response(201, json={"id": COLL, "name": "Uploads",
                                               "kind": "general", "data_residency": "UK",
                                               "created_at": "2026-07-06T00:00:00Z"})
    )
    respx.post(f"{BASE}/vault/documents").mock(return_value=_mint())
    respx.put(UPLOAD_URL).mock(return_value=httpx.Response(200))
    r = runner.invoke(app, ["upload", str(doc), "--collection", "Uploads",
                            "--residency", "uk", "--no-wait"])
    assert r.exit_code == 0, r.stderr
    import json as _json
    body = _json.loads(create.calls.last.request.read())
    assert body == {"name": "Uploads", "kind": "general", "data_residency": "UK"}
