"""Training submit/list happy-path + watch output-purity."""

from __future__ import annotations

import json
import time

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


@respx.mock
def test_submit_returns_run_id(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(200, json={"id": "run_abc"})
    )
    result = runner.invoke(
        app,
        ["training", "submit", "--preset", "deberta-base", "--dataset", "s3://bucket/x.jsonl"],
    )
    assert result.exit_code == 0
    assert "run_abc" in result.stdout
    body = route.calls.last.request.read()
    assert b"deberta-base" in body
    assert "Idempotency-Key" in route.calls.last.request.headers


@respx.mock
def test_list_runs_renders_rows(runner: CliRunner) -> None:
    respx.get(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "run_a",
                        "status": "running",
                        "preset": "deberta-base",
                        "created_at": "2026-05-07",
                    }
                ]
            },
        )
    )
    result = runner.invoke(app, ["training", "list"])
    assert result.exit_code == 0
    assert "run_a" in result.stdout
    assert "running" in result.stdout


@respx.mock
def test_watch_json_stdout_is_single_json_document(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observation #24: with --json, stdout must be EXACTLY one JSON document.

    Progress lines belong on stderr (output.py lock: diagnostics → stderr,
    stdout reserved for the command result) so machine consumers can
    ``gpb --json training watch X | jq .`` without a parse error.
    """
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # no real polling delay
    respx.get(f"{BASE}/training/runs/run_w1").mock(
        side_effect=[
            httpx.Response(200, json={"id": "run_w1", "status": "running", "progress": "40%"}),
            httpx.Response(200, json={"id": "run_w1", "status": "succeeded", "progress": "100%"}),
        ]
    )
    result = runner.invoke(app, ["--json", "training", "watch", "run_w1"])
    assert result.exit_code == 0, result.stderr
    doc = json.loads(result.stdout)  # raises if progress lines pollute stdout
    assert doc["status"] == "succeeded"
    # Progress remains visible to humans — on stderr, not stdout.
    assert "status=running" in result.stderr
    assert "status=succeeded" in result.stderr


@respx.mock
def test_watch_plain_progress_stays_on_stdout(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-JSON watch behaviour is unchanged: progress lines on stdout."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    respx.get(f"{BASE}/training/runs/run_w2").mock(
        side_effect=[
            httpx.Response(200, json={"id": "run_w2", "status": "running", "progress": "40%"}),
            httpx.Response(200, json={"id": "run_w2", "status": "succeeded", "progress": "100%"}),
        ]
    )
    result = runner.invoke(app, ["training", "watch", "run_w2"])
    assert result.exit_code == 0, result.stderr
    # Byte-identical contract (Codex review): exactly the old stdout, no
    # duplication to stderr, no extra lines.
    assert result.stdout == (
        "run_w2 status=running progress=40%\nrun_w2 status=succeeded progress=100%\n"
    )
    assert result.stderr == ""


@respx.mock
def test_watch_json_failed_run_exits_nonzero_with_clean_stdout(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure path: exit 1 preserved AND stdout still a single parseable doc."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    respx.get(f"{BASE}/training/runs/run_w3").mock(
        side_effect=[
            httpx.Response(200, json={"id": "run_w3", "status": "failed", "progress": "12%"}),
        ]
    )
    result = runner.invoke(app, ["--json", "training", "watch", "run_w3"])
    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["status"] == "failed"
