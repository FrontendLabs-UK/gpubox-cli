"""`gpb vault ...` — GPUB-403 path + request/response-shape regression tests.

The whole point of these tests is to catch path drift: every subcommand must
hit the CORRECT gateway route (no `/vault` infix, no double `/v1`) with the
correct request body shape. They mock the HTTP layer with respx and assert on
the exact URL respx matched + the captured request body.

Cross-reference for the asserted shapes:
  * app/vault.py            — /v1/conversations[/...], /v1/conversations/search
  * app/vault_rag/router.py — /v1/corpora[/...], /v1/corpora/upload

BASE already ends in `/v1` (cfg.DEFAULT_API_URL), so respx routes are written
as f"{BASE}/conversations" → https://api.gpubox.ai/v1/conversations, which is
exactly the URL httpx produces from base_url + a leading-slash path.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app

BASE = cfg.DEFAULT_API_URL


@pytest.fixture(autouse=True)
def authed(fake_api_key: str) -> None:
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key, base_url=BASE))


# ---------------------------------------------------------------------------
# search → POST /v1/conversations/search  (NOT /v1/vault/search)
# ---------------------------------------------------------------------------


@respx.mock
def test_search_hits_conversations_search_path_with_mode(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/conversations/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "conversation_id": "11111111-1111-1111-1111-111111111111",
                        "message_id": "22222222-2222-2222-2222-222222222222",
                        "sequence_num": 3,
                        "role": "user",
                        "snippet": "the quick brown fox",
                        "rank": 0.42,
                        "created_at": "2026-06-13T00:00:00Z",
                    }
                ],
            },
        )
    )
    result = runner.invoke(app, ["vault", "search", "brown fox"])
    assert result.exit_code == 0, result.stderr
    # The corrected path was hit — the old /vault/search would 404 here.
    assert route.called
    body = json.loads(route.calls.last.request.read())
    assert body == {"query": "brown fox", "mode": "fts", "limit": 20}
    # Renders the snippet + conversation_id, not the old id/score shape.
    assert "the quick brown fox" in result.stdout
    assert "11111111-1111-1111-1111-111111111111" in result.stdout


@respx.mock
def test_search_substring_mode_passthrough(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/conversations/search").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": []})
    )
    result = runner.invoke(
        app, ["vault", "search", "order_47K", "--mode", "substring", "-n", "5"]
    )
    assert result.exit_code == 0, result.stderr
    body = json.loads(route.calls.last.request.read())
    assert body == {"query": "order_47K", "mode": "substring", "limit": 5}
    assert "(no matches)" in result.stdout


@respx.mock
def test_search_does_not_hit_legacy_vault_path(runner: CliRunner) -> None:
    """Belt-and-braces: the old bogus /v1/vault/search must NOT be called."""
    legacy = respx.post(f"{BASE}/vault/search").mock(
        return_value=httpx.Response(404, json={"error": {"message": "nope"}})
    )
    respx.post(f"{BASE}/conversations/search").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": []})
    )
    result = runner.invoke(app, ["vault", "search", "x"])
    assert result.exit_code == 0, result.stderr
    assert not legacy.called


# ---------------------------------------------------------------------------
# conversations list → GET /v1/conversations
# ---------------------------------------------------------------------------


@respx.mock
def test_conversations_list_path_and_render(runner: CliRunner) -> None:
    route = respx.get(f"{BASE}/conversations").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                        "name": "Tax question",
                        "message_count": 4,
                        "updated_at": "2026-06-12T00:00:00Z",
                        "last_message_at": "2026-06-13T00:00:00Z",
                    }
                ],
                "has_more": False,
            },
        )
    )
    result = runner.invoke(app, ["vault", "conversations", "list", "-n", "7"])
    assert result.exit_code == 0, result.stderr
    assert route.called
    assert route.calls.last.request.url.params["limit"] == "7"
    assert "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" in result.stdout
    assert "msgs=4" in result.stdout
    assert "Tax question" in result.stdout


# ---------------------------------------------------------------------------
# conversations get → GET /v1/conversations/{id} + /messages
# ---------------------------------------------------------------------------


@respx.mock
def test_conversations_get_fetches_meta_and_messages(runner: CliRunner) -> None:
    cid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    meta_route = respx.get(f"{BASE}/conversations/{cid}").mock(
        return_value=httpx.Response(
            200,
            json={"id": cid, "name": "Hello", "message_count": 2},
        )
    )
    msgs_route = respx.get(f"{BASE}/conversations/{cid}/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"role": "user", "content": "hi there"},
                    {"role": "assistant", "content": "hello back"},
                ],
            },
        )
    )
    result = runner.invoke(app, ["vault", "conversations", "get", cid])
    assert result.exit_code == 0, result.stderr
    assert meta_route.called
    assert msgs_route.called
    # order=asc passed to the messages route
    assert msgs_route.calls.last.request.url.params["order"] == "asc"
    assert "[user] hi there" in result.stdout
    assert "[assistant] hello back" in result.stdout


@respx.mock
def test_conversations_get_json_is_single_document(runner: CliRunner) -> None:
    cid = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    respx.get(f"{BASE}/conversations/{cid}").mock(
        return_value=httpx.Response(200, json={"id": cid, "name": "X", "message_count": 0})
    )
    respx.get(f"{BASE}/conversations/{cid}/messages").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": []})
    )
    result = runner.invoke(app, ["--json", "vault", "conversations", "get", cid])
    assert result.exit_code == 0, result.stderr
    doc = json.loads(result.stdout)  # raises if stdout isn't exactly one doc
    assert doc["conversation"]["id"] == cid
    assert doc["messages"]["data"] == []


# ---------------------------------------------------------------------------
# conversations delete → DELETE /v1/conversations/{id}
# ---------------------------------------------------------------------------


@respx.mock
def test_conversations_delete_path(runner: CliRunner) -> None:
    cid = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    route = respx.delete(f"{BASE}/conversations/{cid}").mock(
        return_value=httpx.Response(204)
    )
    result = runner.invoke(app, ["vault", "conversations", "delete", cid])
    assert result.exit_code == 0, result.stderr
    assert route.called
    assert f"deleted conversation {cid}" in result.stdout


# ---------------------------------------------------------------------------
# corpora list → GET /v1/corpora
# ---------------------------------------------------------------------------


@respx.mock
def test_corpora_list_path_and_render(runner: CliRunner) -> None:
    route = respx.get(f"{BASE}/corpora").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                        "name": "Handbook",
                        "chunk_count": 12,
                    }
                ],
            },
        )
    )
    result = runner.invoke(app, ["vault", "corpora", "list"])
    assert result.exit_code == 0, result.stderr
    assert route.called
    assert "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee" in result.stdout
    assert "Handbook" in result.stdout
    assert "chunks=12" in result.stdout


# ---------------------------------------------------------------------------
# corpora get → GET /v1/corpora/{id}
# ---------------------------------------------------------------------------


@respx.mock
def test_corpora_get_path(runner: CliRunner) -> None:
    cid = "ffffffff-ffff-ffff-ffff-ffffffffffff"
    route = respx.get(f"{BASE}/corpora/{cid}").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": cid,
                "name": "Docs",
                "chunk_count": 3,
                "embedded_chunk_count": 3,
                "total_bytes": 999,
            },
        )
    )
    result = runner.invoke(app, ["vault", "corpora", "get", cid])
    assert result.exit_code == 0, result.stderr
    assert route.called
    assert cid in result.stdout
    assert "chunks=3" in result.stdout


# ---------------------------------------------------------------------------
# corpora delete → DELETE /v1/corpora/{id}
# ---------------------------------------------------------------------------


@respx.mock
def test_corpora_delete_path(runner: CliRunner) -> None:
    cid = "10101010-1010-1010-1010-101010101010"
    route = respx.delete(f"{BASE}/corpora/{cid}").mock(
        return_value=httpx.Response(204)
    )
    result = runner.invoke(app, ["vault", "corpora", "delete", cid])
    assert result.exit_code == 0, result.stderr
    assert route.called
    assert f"deleted corpus {cid}" in result.stdout


# ---------------------------------------------------------------------------
# corpora create (JSON) → POST /v1/corpora with {name, source_type, content}
# ---------------------------------------------------------------------------


@respx.mock
def test_corpora_create_json_body_shape(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/corpora").mock(
        return_value=httpx.Response(
            201, json={"id": "20202020-2020-2020-2020-202020202020", "name": "KB"}
        )
    )
    result = runner.invoke(
        app,
        [
            "vault",
            "corpora",
            "create",
            "--name",
            "KB",
            "--source-type",
            "manual",
            "--content",
            "some text",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert route.called
    body = json.loads(route.calls.last.request.read())
    assert body == {"name": "KB", "source_type": "manual", "content": "some text"}
    # Idempotency-Key is emitted for the create POST (matches training submit).
    assert "Idempotency-Key" in route.calls.last.request.headers
    assert "20202020-2020-2020-2020-202020202020" in result.stdout


@respx.mock
def test_corpora_create_url_source(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/corpora").mock(
        return_value=httpx.Response(201, json={"id": "30303030-3030-3030-3030-303030303030"})
    )
    result = runner.invoke(
        app,
        [
            "vault",
            "corpora",
            "create",
            "--name",
            "Site",
            "--source-type",
            "url",
            "--content",
            "https://example.com/doc",
        ],
    )
    assert result.exit_code == 0, result.stderr
    body = json.loads(route.calls.last.request.read())
    assert body["source_type"] == "url"
    assert body["content"] == "https://example.com/doc"


# ---------------------------------------------------------------------------
# corpora create (PDF) → multipart POST /v1/corpora/upload
# ---------------------------------------------------------------------------


@respx.mock
def test_corpora_create_from_file_uses_upload_path(
    runner: CliRunner, tmp_path
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    route = respx.post(f"{BASE}/corpora/upload").mock(
        return_value=httpx.Response(
            201, json={"id": "40404040-4040-4040-4040-404040404040", "name": "Doc"}
        )
    )
    result = runner.invoke(
        app,
        ["vault", "corpora", "create", "--name", "Doc", "--from-file", str(pdf)],
    )
    assert result.exit_code == 0, result.stderr
    assert route.called
    # multipart form carries the name field + the file part.
    req_body = route.calls.last.request.read()
    assert b"doc.pdf" in req_body
    assert b"Doc" in req_body
    assert "40404040-4040-4040-4040-404040404040" in result.stdout


# ---------------------------------------------------------------------------
# enable / disable — operator-only, NO network call (no public route exists)
# ---------------------------------------------------------------------------


@respx.mock
def test_enable_makes_no_network_call(runner: CliRunner) -> None:
    # Register the legacy route so we can prove it is never hit.
    legacy = respx.post(f"{BASE}/vault/enable").mock(
        return_value=httpx.Response(404)
    )
    result = runner.invoke(app, ["vault", "enable"])
    assert result.exit_code == 0, result.stderr
    assert not legacy.called
    assert "operator-only" in result.stdout
    assert "support@gpubox.ai" in result.stdout


@respx.mock
def test_disable_makes_no_network_call(runner: CliRunner) -> None:
    legacy = respx.post(f"{BASE}/vault/enable").mock(
        return_value=httpx.Response(404)
    )
    result = runner.invoke(app, ["vault", "disable"])
    assert result.exit_code == 0, result.stderr
    assert not legacy.called
    assert "operator-only" in result.stdout


@respx.mock
def test_enable_json_mode_is_structured(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--json", "vault", "enable"])
    assert result.exit_code == 0, result.stderr
    doc = json.loads(result.stdout)
    assert doc["available_via_cli"] is False
    assert doc["contact"] == "support@gpubox.ai"
