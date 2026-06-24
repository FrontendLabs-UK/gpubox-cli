"""`gpb vault enrich ...` — path + request/response-shape + gating tests.

The enrich command wraps the gateway's web-enrichment surface:
  * preview -> POST /v1/vault/enrich        (writes nothing; the default)
  * --save  -> POST /v1/vault/enrich        (preview) then
               POST /v1/vault/enrich/save   (persist the ok records)

BASE already ends in `/v1`, so the command writes leading-slash paths WITHOUT
the `/v1` prefix and WITHOUT a `/vault` infix-error (cross-reference
app/vault_enrich/rest.py + app/vault_hardened/routes.py in the gateway repo).
We mock the HTTP layer with respx and assert the exact URL + request body, and
that the documented gating errors (save_gated / save_not_enabled) surface
cleanly (non-zero exit, message on stderr) rather than crashing.
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


def _record(url: str, status: str = "ok", **over) -> dict:
    rec = {
        "url": url,
        "extracted": {"price": "£10/mo"},
        "fetched_at": "2026-06-24T08:40:12.345678+00:00",
        "status": status,
        "source_host": "example.com",
        "truncated": False,
        "pii_detected": False,
        "pii_categories": {},
        "pii_redacted_fields": [],
        "coverage": 1.0,
        "needs_disambiguation": False,
        "reason": None,
        "injection_suspected": False,
        "prompt_version": "ve-prompts-1.1.0",
        "extractor_version": "ve-extractor-1.1.0",
    }
    rec.update(over)
    return rec


# ---------------------------------------------------------------------------
# preview → POST /v1/vault/enrich  (the default, no /save)
# ---------------------------------------------------------------------------


@respx.mock
def test_preview_hits_enrich_path_with_body(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/vault/enrich").mock(
        return_value=httpx.Response(
            200,
            json={
                "records": [_record("https://example.com/pricing")],
                "saved": False,
                "save_status": "VE-W2: saving to the vault lands after the DB cutover.",
            },
        )
    )
    result = runner.invoke(
        app,
        ["vault", "enrich", "https://example.com/pricing", "--capture", "pricing"],
    )
    assert result.exit_code == 0, result.stderr
    assert route.called
    body = json.loads(route.calls.last.request.read())
    assert body["urls"] == ["https://example.com/pricing"]
    assert body["capture"] == "pricing"
    assert body["respect_robots"] is True
    # No save route was hit on the preview path.
    assert result.stdout  # rendered something


@respx.mock
def test_preview_json_mode_emits_single_document(runner: CliRunner) -> None:
    respx.post(f"{BASE}/vault/enrich").mock(
        return_value=httpx.Response(
            200,
            json={"records": [_record("https://x.test")], "saved": False, "save_status": "s"},
        )
    )
    result = runner.invoke(
        app, ["--json", "vault", "enrich", "https://x.test", "--capture", "x"]
    )
    assert result.exit_code == 0, result.stderr
    doc = json.loads(result.stdout)  # stdout must be one parseable JSON doc
    assert doc["records"][0]["pii_detected"] is False
    assert doc["records"][0]["extractor_version"] == "ve-extractor-1.1.0"


@respx.mock
def test_preview_passes_schema_when_given(runner: CliRunner, tmp_path) -> None:
    schema_file = tmp_path / "s.json"
    schema_file.write_text(json.dumps({"type": "object", "properties": {"price": {}}}))
    route = respx.post(f"{BASE}/vault/enrich").mock(
        return_value=httpx.Response(200, json={"records": [], "saved": False, "save_status": "s"})
    )
    result = runner.invoke(
        app,
        ["vault", "enrich", "https://x.test", "--capture", "x", "--schema", str(schema_file)],
    )
    assert result.exit_code == 0, result.stderr
    body = json.loads(route.calls.last.request.read())
    assert body["schema"] == {"type": "object", "properties": {"price": {}}}


@respx.mock
def test_batch_reads_urls_from_file(runner: CliRunner, tmp_path) -> None:
    batch = tmp_path / "urls.txt"
    batch.write_text("# a comment\nhttps://a.test\n\nhttps://b.test\n")
    route = respx.post(f"{BASE}/vault/enrich").mock(
        return_value=httpx.Response(
            200,
            json={"records": [_record("https://a.test"), _record("https://b.test")], "saved": False, "save_status": "s"},
        )
    )
    result = runner.invoke(
        app, ["vault", "enrich", "--batch", str(batch), "--capture", "pricing"]
    )
    assert result.exit_code == 0, result.stderr
    body = json.loads(route.calls.last.request.read())
    # Comment + blank line stripped; both URLs forwarded.
    assert body["urls"] == ["https://a.test", "https://b.test"]


def test_no_url_and_no_batch_errors(runner: CliRunner) -> None:
    result = runner.invoke(app, ["vault", "enrich", "--capture", "x"])
    assert result.exit_code != 0
    assert "URL" in result.stderr or "batch" in result.stderr


# ---------------------------------------------------------------------------
# --save → preview then POST /v1/vault/enrich/save
# ---------------------------------------------------------------------------


@respx.mock
def test_save_previews_then_saves_ok_records(runner: CliRunner) -> None:
    preview = respx.post(f"{BASE}/vault/enrich").mock(
        return_value=httpx.Response(
            200,
            json={
                "records": [
                    _record("https://ok.test", status="ok"),
                    _record("https://bad.test", status="blocked", extracted={}),
                ],
                "saved": False,
                "save_status": "s",
            },
        )
    )
    save = respx.post(f"{BASE}/vault/enrich/save").mock(
        return_value=httpx.Response(
            200,
            json={"saved": True, "document_ids": ["doc-1"], "skipped": []},
        )
    )
    result = runner.invoke(
        app,
        ["vault", "enrich", "https://ok.test", "--capture", "pricing", "--save", "--collection", "col-1"],
    )
    assert result.exit_code == 0, result.stderr
    assert preview.called and save.called
    save_body = json.loads(save.calls.last.request.read())
    # Only the ok record is forwarded, projected to exactly the save fields.
    assert len(save_body["records"]) == 1
    assert save_body["records"][0]["url"] == "https://ok.test"
    assert set(save_body["records"][0].keys()) == {
        "url",
        "extracted",
        "fetched_at",
        "status",
        "source_host",
        "truncated",
    }
    assert save_body["collection"] == "col-1"
    # The blocked URL never reached /save, so the human output must still call
    # it out as preview-skipped (not silently dropped).
    assert "preview_skipped=1" in result.stdout
    assert "https://bad.test" in result.stdout


@respx.mock
def test_save_json_mode_surfaces_preview_skipped(runner: CliRunner) -> None:
    respx.post(f"{BASE}/vault/enrich").mock(
        return_value=httpx.Response(
            200,
            json={
                "records": [
                    _record("https://ok.test", status="ok"),
                    _record("https://bad.test", status="blocked", extracted={}, reason="SSRF target"),
                ],
                "saved": False,
                "save_status": "s",
            },
        )
    )
    respx.post(f"{BASE}/vault/enrich/save").mock(
        return_value=httpx.Response(200, json={"saved": True, "document_ids": ["doc-1"], "skipped": []})
    )
    result = runner.invoke(
        app, ["--json", "vault", "enrich", "https://ok.test", "--capture", "x", "--save"]
    )
    assert result.exit_code == 0, result.stderr
    doc = json.loads(result.stdout)
    assert doc["document_ids"] == ["doc-1"]
    assert len(doc["preview_skipped"]) == 1
    assert doc["preview_skipped"][0]["url"] == "https://bad.test"


def test_collection_without_save_rejected(runner: CliRunner) -> None:
    result = runner.invoke(
        app, ["vault", "enrich", "https://x.test", "--capture", "x", "--collection", "col-1"]
    )
    assert result.exit_code != 0
    assert "collection" in result.stderr.lower()


@respx.mock
def test_save_with_no_ok_records_skips_save(runner: CliRunner) -> None:
    respx.post(f"{BASE}/vault/enrich").mock(
        return_value=httpx.Response(
            200,
            json={
                "records": [_record("https://bad.test", status="blocked", extracted={})],
                "saved": False,
                "save_status": "s",
            },
        )
    )
    save = respx.post(f"{BASE}/vault/enrich/save").mock(
        return_value=httpx.Response(200, json={"saved": True, "document_ids": [], "skipped": []})
    )
    result = runner.invoke(
        app, ["vault", "enrich", "https://bad.test", "--capture", "x", "--save"]
    )
    assert result.exit_code == 0, result.stderr
    # No ok records -> the save POST is never made.
    assert not save.called


# ---------------------------------------------------------------------------
# gating: the documented error codes surface cleanly (no crash)
# ---------------------------------------------------------------------------


@respx.mock
def test_save_gated_on_preview_route_surfaces_cleanly(runner: CliRunner) -> None:
    """If the gateway 400s save_gated on the preview route, the CLI exits
    non-zero with the message on stderr (not a traceback)."""
    respx.post(f"{BASE}/vault/enrich").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": "save_gated",
                    "message": "Saving to the vault is gated on VE-W2 (the DB cutover). Use preview.",
                    "param": "save",
                    "request_id": None,
                }
            },
        )
    )
    result = runner.invoke(app, ["vault", "enrich", "https://x.test", "--capture", "x"])
    assert result.exit_code != 0
    assert "gated" in result.stderr.lower()


@respx.mock
def test_save_not_enabled_surfaces_cleanly(runner: CliRunner) -> None:
    """The save route 404s save_not_enabled until the flag flips; the CLI must
    surface it cleanly after a successful preview, never crash."""
    respx.post(f"{BASE}/vault/enrich").mock(
        return_value=httpx.Response(
            200,
            json={"records": [_record("https://ok.test")], "saved": False, "save_status": "s"},
        )
    )
    respx.post(f"{BASE}/vault/enrich/save").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": "save_not_enabled",
                    "message": "vault web-enrichment save is not enabled",
                    "param": None,
                    "request_id": None,
                }
            },
        )
    )
    result = runner.invoke(
        app, ["vault", "enrich", "https://ok.test", "--capture", "x", "--save"]
    )
    assert result.exit_code != 0
    assert "not enabled" in result.stderr.lower()
