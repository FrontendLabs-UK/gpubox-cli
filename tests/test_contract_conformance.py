"""CI contract-conformance guard.

For every mutating CLI command we drive the REAL command through `runner.invoke`
against a respx-mocked transport, capture the EXACT request the command code
builds (method, path, JSON body — the literal bytes httpx serialised), and assert
it conforms to the pinned gateway OpenAPI snapshot via
`tests.contract.conformance.assert_request_conforms`.

Why capture at the transport layer rather than hand-write a request manifest:
a hand-written manifest rots independently of the command code. respx intercepts
the request the command ACTUALLY sends, so this file asserts against live command
behaviour — if a command's body drifts, this catches it without anyone updating a
parallel list.

Coverage: every JSON-bodied POST/PUT/PATCH and every mutating DELETE across the
CLI (assistants, hosting, finetune, training, argus, search, workspace, billing,
embed, chat, users, vault — including vault search, conversation delete, and
`finetune use --clear`). The two multipart commands (transcribe,
`vault corpora create --from-file`) build `multipart/form-data` bodies the JSON
validator cannot inspect, so each is driven LIVE through respx and its captured
request's (method, path) is asserted to resolve to a registered operation
(`test_transcribe_route_conforms`, `test_vault_corpora_upload_route_conforms`) —
catching a path/method drift on an upload command even though the field shape is
out of scope. Pure GET reads need no body-shape assertion — the one exception is
`training download`, which is GET but is exactly the route the historical
`/artifact` drift broke, so it gets an explicit registered-route assertion. The
negative guard-proof — that this harness actually rejects the eight historical
PR-#12 drifts — lives in `test_contract_guard_proof.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app
from tests.contract.conformance import (
    MISSING,
    assert_request_conforms,
    load_spec,
    resolve_operation,
)

BASE = cfg.DEFAULT_API_URL

# One shared pinned snapshot for the whole module.
SPEC = load_spec()


@pytest.fixture(autouse=True)
def authed(fake_api_key: str) -> None:
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key, base_url=BASE))


def _assert_request_conforms(request: httpx.Request) -> None:
    """Extract (method, path, body) from a captured httpx request and validate.

    `request.url.path` is the full path INCLUDING the base_url's `/v1` prefix
    (e.g. `/v1/assistants/abc`) — exactly the form the spec stores, so the
    validator's verbatim-first match resolves it directly.

    No request body -> pass MISSING (NOT None), so the validator can tell "no
    body sent" apart from a JSON `null`/`{}` body the gateway would still see.
    """
    raw = request.read()
    body = json.loads(raw) if raw else MISSING
    assert_request_conforms(SPEC, request.method, request.url.path, body)


def _conforms(route) -> None:
    """Assert the LAST request captured by a respx route conforms."""
    assert route.called, "command never sent the expected request"
    _assert_request_conforms(route.calls.last.request)


def _pin_workspace(ws: str) -> None:
    settings = cfg.load_settings()
    settings.extra["active_workspace"] = ws
    cfg.save_settings(settings)


# ---------------------------------------------------------------------------
# assistants  (PR #12: create slug, update POST-not-PATCH, run /chat/completions)
# ---------------------------------------------------------------------------


@respx.mock
def test_assistants_create_conforms(runner: CliRunner, tmp_path: Path) -> None:
    prompt = tmp_path / "p.txt"
    prompt.write_text("be helpful", encoding="utf-8")
    route = respx.post(f"{BASE}/assistants").mock(
        return_value=httpx.Response(200, json={"id": "asst_1"})
    )
    res = runner.invoke(
        app,
        ["assistants", "create", "--slug", "support", "--name", "S",
         "--instructions", str(prompt), "--model", "qwen2.5-32b-instruct"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_assistants_update_conforms(runner: CliRunner, tmp_path: Path) -> None:
    prompt = tmp_path / "p.txt"
    prompt.write_text("updated", encoding="utf-8")
    route = respx.post(f"{BASE}/assistants/asst_9").mock(
        return_value=httpx.Response(200, json={"id": "asst_9"})
    )
    res = runner.invoke(
        app, ["assistants", "update", "asst_9", "--name", "New",
              "--instructions", str(prompt)]
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_assistants_run_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "hi"}}]}
        )
    )
    res = runner.invoke(app, ["assistants", "run", "asst_42", "say hi"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_assistants_delete_conforms(runner: CliRunner) -> None:
    route = respx.delete(f"{BASE}/assistants/asst_9").mock(
        return_value=httpx.Response(204)
    )
    res = runner.invoke(app, ["assistants", "delete", "asst_9"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# hosting  (PR #12: promote key rename + --name, tier -> /transition POST)
# ---------------------------------------------------------------------------


@respx.mock
def test_hosting_promote_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/hosting/models").mock(
        return_value=httpx.Response(201, json={"id": "model_x"})
    )
    res = runner.invoke(
        app, ["hosting", "promote", "run_abc", "--name", "acme-v1", "--tier", "warm"]
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_hosting_tier_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/hosting/models/model_x/transition").mock(
        return_value=httpx.Response(200, json={"id": "model_x"})
    )
    res = runner.invoke(app, ["hosting", "tier", "model_x", "--tier", "always_hot"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_hosting_delete_conforms(runner: CliRunner) -> None:
    route = respx.delete(f"{BASE}/hosting/models/model_x").mock(
        return_value=httpx.Response(204)
    )
    res = runner.invoke(app, ["hosting", "delete", "model_x"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# finetune  (PR #12: GPUB-458 create body — preset + hyperparams, no flat keys)
# ---------------------------------------------------------------------------


@respx.mock
def test_finetune_create_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(200, json={"id": "run-1"})
    )
    res = runner.invoke(
        app,
        ["finetune", "create", "--preset", "qwen32b-lora-r16",
         "--since", "2026-01-01T00:00:00Z", "--until", "2026-02-01T00:00:00Z",
         "--epochs", "2", "--batch-size", "4", "--learning-rate", "0.0001"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_finetune_use_conforms(runner: CliRunner) -> None:
    route = respx.put(f"{BASE}/finetune/active").mock(
        return_value=httpx.Response(200, json={"hosted_model_name": "acme-v1"})
    )
    res = runner.invoke(app, ["finetune", "use", "acme-v1"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# training  (PR #12: submit GPUB-458 body)
# ---------------------------------------------------------------------------


@respx.mock
def test_training_submit_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(200, json={"id": "run_abc"})
    )
    res = runner.invoke(
        app,
        ["training", "submit", "--preset", "deberta-base",
         "--since", "2026-01-01T00:00:00Z", "--epochs", "3"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_training_submit_intensity_conforms(runner: CliRunner) -> None:
    """GPUB-620: `--intensity` sends top-level training_intensity — it must be a
    declared field in the pinned contract (this catches a stale pin)."""
    route = respx.post(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(200, json={"id": "run_i"})
    )
    res = runner.invoke(
        app,
        ["training", "submit", "--preset", "qwen32b-lora-r16", "--intensity", "standard"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_finetune_create_intensity_conforms(runner: CliRunner) -> None:
    """GPUB-620: same training_intensity field via `gpb finetune create`."""
    route = respx.post(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(200, json={"id": "run_i"})
    )
    res = runner.invoke(
        app,
        ["finetune", "create", "--preset", "qwen32b-lora-r16", "--intensity", "thorough"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_training_cancel_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/training/runs/run_abc/cancel").mock(
        return_value=httpx.Response(200, json={"id": "run_abc", "status": "cancelled"})
    )
    res = runner.invoke(app, ["training", "cancel", "run_abc"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# argus  (create agent, run agent, mark inbox item read, delete agent)
# ---------------------------------------------------------------------------


@respx.mock
def test_argus_create_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/argus/agents").mock(
        return_value=httpx.Response(200, json={"id": "agent_1"})
    )
    res = runner.invoke(
        app,
        ["argus", "create", "--question", "track CUDA CVEs",
         "--doc", "doc-1", "--cadence", "daily"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_argus_run_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/argus/agents/agent_1/run").mock(
        return_value=httpx.Response(202, json={"status": "queued"})
    )
    res = runner.invoke(app, ["argus", "run", "agent_1"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_argus_inbox_read_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/argus/inbox/item_1/read").mock(
        return_value=httpx.Response(200, json={"id": "item_1", "read": True})
    )
    res = runner.invoke(app, ["argus", "read", "item_1"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_argus_delete_conforms(runner: CliRunner) -> None:
    route = respx.delete(f"{BASE}/argus/agents/agent_1").mock(
        return_value=httpx.Response(204)
    )
    res = runner.invoke(app, ["argus", "delete", "agent_1"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@respx.mock
def test_search_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"results": [], "answer": ""})
    )
    res = runner.invoke(
        app, ["search", "what changed in Q2", "--synthesize", "--sources", "docs,chat"]
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


@respx.mock
def test_workspace_create_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/workspaces").mock(
        return_value=httpx.Response(201, json={"id": "ws_1"})
    )
    res = runner.invoke(
        app,
        ["workspace", "create", "--name", "Acme", "--slug", "acme",
         "--description", "team space"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_workspace_delete_conforms(runner: CliRunner) -> None:
    route = respx.delete(f"{BASE}/workspaces/ws_1").mock(
        return_value=httpx.Response(204)
    )
    res = runner.invoke(app, ["workspace", "delete", "ws_1"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_workspace_update_conforms(runner: CliRunner) -> None:
    route = respx.patch(f"{BASE}/workspaces/ws_1").mock(
        return_value=httpx.Response(200, json={"id": "ws_1", "name": "Renamed"})
    )
    res = runner.invoke(
        app,
        ["workspace", "update", "ws_1", "--name", "Renamed",
         "--default-model", "qwen2.5-32b-instruct",
         "--response-language", "en-GB", "--watch-cadence", "daily"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_auth_set_name_conforms(runner: CliRunner) -> None:
    route = respx.patch(f"{BASE}/auth/me").mock(
        return_value=httpx.Response(200, json={"display_name": "Ada"})
    )
    res = runner.invoke(app, ["auth", "set-name", "Ada"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# billing  (Stripe GBP + Paystack NGN checkout)
# ---------------------------------------------------------------------------


@respx.mock
def test_billing_topup_gbp_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/billing/checkout-sessions").mock(
        return_value=httpx.Response(200, json={"url": "https://pay.example/x"})
    )
    res = runner.invoke(app, ["billing", "topup", "--amount-gbp", "20"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_billing_topup_ngn_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/billing/paystack/initialize").mock(
        return_value=httpx.Response(
            200, json={"authorization_url": "https://paystack.example/x"}
        )
    )
    res = runner.invoke(app, ["billing", "topup", "--amount-ngn", "30000"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


@respx.mock
def test_embed_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200, json={"data": [{"embedding": [0.1, 0.2]}]}
        )
    )
    res = runner.invoke(app, ["embed", "hello world"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "hi back"}}]}
        )
    )
    res = runner.invoke(app, ["--json", "chat", "say hi", "--model", "qwen2.5-32b-instruct"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# users  (invite user, register oidc client)
# ---------------------------------------------------------------------------


@respx.mock
def test_users_invite_conforms(runner: CliRunner) -> None:
    tenant = "11111111-1111-1111-1111-111111111111"
    route = respx.post(f"{BASE}/tenants/{tenant}/users").mock(
        return_value=httpx.Response(200, json={"id": "user_1"})
    )
    res = runner.invoke(
        app,
        ["users", "invite", "teammate@example.com", "--role", "editor",
         "--name", "Tee", "--tenant", tenant],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_users_create_client_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/oidc/clients").mock(
        return_value=httpx.Response(200, json={"client_id": "c_1"})
    )
    res = runner.invoke(
        app,
        ["users", "oidc", "create", "--name", "My App",
         "--redirect-uri", "https://app.example/callback"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# vault  (corpora JSON create path)
# ---------------------------------------------------------------------------


@respx.mock
def test_vault_corpora_create_conforms(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/corpora").mock(
        return_value=httpx.Response(201, json={"id": "corpus_1"})
    )
    res = runner.invoke(
        app,
        ["vault", "corpora", "create", "--name", "kb",
         "--source-type", "manual", "--content", "some text"],
    )
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_vault_search_conforms(runner: CliRunner) -> None:
    """POST /conversations/search (Postgres FTS) — app__vault__SearchRequest is
    extra='forbid' with required `query`."""
    route = respx.post(f"{BASE}/conversations/search").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": []})
    )
    res = runner.invoke(app, ["vault", "search", "deploy notes", "--mode", "fts"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


@respx.mock
def test_vault_conversation_delete_conforms(runner: CliRunner) -> None:
    route = respx.delete(f"{BASE}/conversations/conv_1").mock(
        return_value=httpx.Response(204)
    )
    res = runner.invoke(app, ["vault", "conversations", "delete", "conv_1"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# finetune use --clear  (DELETE /finetune/active — a no-body mutating verb the
# happy-path `use` test does not cover)
# ---------------------------------------------------------------------------


@respx.mock
def test_finetune_use_clear_conforms(runner: CliRunner) -> None:
    route = respx.delete(f"{BASE}/finetune/active").mock(
        return_value=httpx.Response(200, json={"hosted_model_name": None})
    )
    res = runner.invoke(app, ["finetune", "use", "ignored", "--clear"])
    assert res.exit_code == 0, res.stderr
    _conforms(route)


# ---------------------------------------------------------------------------
# training download  (GET /training/runs/{id}/download signed-URL envelope)
#
# A GET, but it is exactly the route the historical `/artifact` drift broke
# (404). The guard-proof rejects the static `/artifact` tuple; THIS test proves
# the live command hits the REGISTERED `/download` route — closing the loop the
# guard-proof alone cannot (Codex finding).
# ---------------------------------------------------------------------------


@respx.mock
def test_training_download_route_is_registered(runner: CliRunner, tmp_path: Path) -> None:
    import hashlib

    payload = b"adapter-bytes"
    sha = hashlib.sha256(payload).hexdigest()
    env_route = respx.get(f"{BASE}/training/runs/run_d/download").mock(
        return_value=httpx.Response(
            200,
            json={"url": "https://r2.example/signed/run_d.bin", "sha256": sha,
                  "expires_in": 3600, "size_bytes": len(payload)},
        )
    )
    respx.get("https://r2.example/signed/run_d.bin").mock(
        return_value=httpx.Response(200, content=payload)
    )
    dest = tmp_path / "adapter.bin"
    res = runner.invoke(app, ["training", "download", "run_d", str(dest)])
    assert res.exit_code == 0, res.stderr
    # The gateway-side request (not the signed-URL blob fetch) must conform.
    _conforms(env_route)


# ---------------------------------------------------------------------------
# Multipart uploads — transcribe and `vault corpora create --from-file` send
# multipart/form-data, NOT application/json, so the JSON request-body validator
# cannot inspect their FIELD shape. But the method+path is still a drift surface
# (a wrong path/method here is the same 404/405 class), so we drive the LIVE
# command through respx and assert the request it actually sends resolves to a
# registered operation. (Codex R2: a static spec check would pass even if the
# command drifted to a wrong path — these tests exercise the real command.)
# ---------------------------------------------------------------------------


def _assert_route_registered(request: httpx.Request) -> None:
    """Assert (method, path) of a captured request resolves to a real operation.

    Field-shape is intentionally not validated (multipart body) — this is the
    route+method existence half of conformance, which is what catches a wrong
    path/method drift on an upload command.
    """
    op = resolve_operation(SPEC, request.method, request.url.path)
    assert op is not None, (
        f"unregistered method/path: {request.method} {request.url.path} "
        f"has no operation in the pinned gateway contract"
    )


@respx.mock
def test_transcribe_route_conforms(runner: CliRunner, tmp_path: Path) -> None:
    """gpb transcribe -> multipart POST /audio/transcriptions (registered)."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFfake-wav-bytes")
    route = respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello"})
    )
    res = runner.invoke(app, ["transcribe", str(audio), "--format", "json"])
    assert res.exit_code == 0, res.stderr
    assert route.called, "transcribe never sent the expected request"
    _assert_route_registered(route.calls.last.request)


@respx.mock
def test_transcribe_owned_model_id_sent(runner: CliRunner, tmp_path: Path) -> None:
    """gpb transcribe --model ng-whisper-medium-v4b forwards that model id.

    The owned-model STT lane (gateway feat/stt-medium-v4b-route) is selected
    by the `model` form field; the CLI must pass the operator-chosen id
    through unchanged so the gateway can route it to the owned upstream.
    """
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFfake-wav-bytes")
    route = respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello"})
    )
    res = runner.invoke(
        app,
        ["transcribe", str(audio), "--model", "ng-whisper-medium-v4b",
         "--format", "json"],
    )
    assert res.exit_code == 0, res.stderr
    assert route.called, "transcribe never sent the expected request"
    sent = route.calls.last.request
    _assert_route_registered(sent)
    # Multipart body carries the chosen model id verbatim.
    assert b"ng-whisper-medium-v4b" in sent.content
    assert b'name="model"' in sent.content


@respx.mock
def test_vault_corpora_upload_route_conforms(
    runner: CliRunner, tmp_path: Path
) -> None:
    """gpb vault corpora create --from-file -> multipart POST /corpora/upload."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7 fake")
    route = respx.post(f"{BASE}/corpora/upload").mock(
        return_value=httpx.Response(201, json={"id": "corpus_u"})
    )
    res = runner.invoke(
        app,
        ["vault", "corpora", "create", "--name", "kb", "--from-file", str(pdf)],
    )
    assert res.exit_code == 0, res.stderr
    assert route.called, "corpora upload never sent the expected request"
    _assert_route_registered(route.calls.last.request)
