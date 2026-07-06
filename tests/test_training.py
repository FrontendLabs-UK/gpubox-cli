"""Training submit/list happy-path + watch output-purity."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

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
        ["training", "submit", "--preset", "deberta-base"],
    )
    assert result.exit_code == 0
    assert "run_abc" in result.stdout
    sent = json.loads(route.calls.last.request.read())
    # GPUB-458: only `preset` for a plain submit; vault corpus + source='vault'
    # are server-side.
    assert sent == {"preset": "deberta-base"}
    assert "Idempotency-Key" in route.calls.last.request.headers


@respx.mock
def test_submit_nests_overrides_under_hyperparams(runner: CliRunner) -> None:
    """epochs/batch_size/learning_rate go under `hyperparams`; --since/--until
    are top-level vault-window bounds. No legacy flat or dataset keys leak in."""
    route = respx.post(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(200, json={"id": "run_h"})
    )
    result = runner.invoke(
        app,
        [
            "training", "submit", "--preset", "deberta-base",
            "--since", "2026-01-01T00:00:00Z", "--until", "2026-02-01T00:00:00Z",
            "--epochs", "3", "--batch-size", "16", "--learning-rate", "0.00002",
        ],
    )
    assert result.exit_code == 0, result.stderr
    sent = json.loads(route.calls.last.request.read())
    assert sent == {
        "preset": "deberta-base",
        "since": "2026-01-01T00:00:00Z",
        "until": "2026-02-01T00:00:00Z",
        "hyperparams": {"epochs": 3, "batch_size": 16, "learning_rate": 0.00002},
    }
    # Regression: TrainingRunCreate is extra='forbid'.
    for forbidden in ("dataset", "dataset_url", "name", "epochs", "batch_size", "learning_rate"):
        assert forbidden not in sent


@respx.mock
def test_submit_sends_intensity_top_level(runner: CliRunner) -> None:
    """GPUB-620: --intensity is a TOP-LEVEL field (training_intensity), NOT nested
    under hyperparams; it coexists with a preset and passes through verbatim."""
    route = respx.post(f"{BASE}/training/runs").mock(
        return_value=httpx.Response(200, json={"id": "run_i"})
    )
    result = runner.invoke(
        app,
        ["training", "submit", "--preset", "qwen32b-lora-r16", "--intensity", "thorough"],
    )
    assert result.exit_code == 0, result.stderr
    sent = json.loads(route.calls.last.request.read())
    assert sent == {"preset": "qwen32b-lora-r16", "training_intensity": "thorough"}
    assert "hyperparams" not in sent  # not nested


@respx.mock
def test_presets_render_intensities(runner: CliRunner) -> None:
    """GPUB-620: `training presets` shows per-preset default_intensity and the
    intensity catalog (name/steps/lr, default flagged)."""
    respx.get(f"{BASE}/training/presets").mock(
        return_value=httpx.Response(200, json={
            "data": [
                {"name": "qwen32b-lora-r16", "model_base": "Qwen", "min_vram_gb": 24,
                 "estimated_gpu_seconds": 3600, "default_intensity": "standard"},
                {"name": "qwen32b-smoke", "model_base": "Qwen", "min_vram_gb": 24,
                 "estimated_gpu_seconds": 120, "default_intensity": None},
            ],
            "intensities": [
                {"name": "quick", "max_train_steps": 100, "learning_rate": 0.0002, "is_default": False},
                {"name": "standard", "max_train_steps": 250, "learning_rate": 0.0002, "is_default": True},
                {"name": "thorough", "max_train_steps": 500, "learning_rate": 0.0003, "is_default": False},
            ],
        })
    )
    result = runner.invoke(app, ["training", "presets"])
    assert result.exit_code == 0, result.stderr
    assert "default_intensity=standard" in result.stdout
    # a null default_intensity (preset pins its own steps) renders as `none`,
    # not dropped — that's a real signal.
    assert "default_intensity=none" in result.stdout
    assert "intensities (--intensity):" in result.stdout
    assert "standard" in result.stdout and "(default)" in result.stdout
    assert "steps=500" in result.stdout


def test_submit_rejects_dataset_flag(runner: CliRunner) -> None:
    """--dataset was removed (GPUB-458) — usage error, not a gateway 422."""
    result = runner.invoke(
        app,
        ["training", "submit", "--preset", "p", "--dataset", "s3://bucket/x.jsonl"],
    )
    assert result.exit_code != 0


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


# ---------------------------------------------------------------------------
# download — JSON signed-URL envelope (GPUB drift fix), not /artifact stream
# ---------------------------------------------------------------------------


@respx.mock
def test_download_uses_signed_url_envelope(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The gateway exposes GET /training/runs/{id}/download returning
    {url, sha256, ...}; the CLI must call THAT (not the dead /artifact route),
    then fetch the signed URL and write the bytes to disk."""
    payload = b"fake-adapter-bytes"
    sha = hashlib.sha256(payload).hexdigest()
    env_route = respx.get(f"{BASE}/training/runs/run_d/download").mock(
        return_value=httpx.Response(
            200,
            json={
                "url": "https://r2.example/signed/run_d.bin?sig=abc",
                "expires_in": 3600,
                "sha256": sha,
                "size_bytes": len(payload),
            },
        )
    )
    blob_route = respx.get("https://r2.example/signed/run_d.bin").mock(
        return_value=httpx.Response(200, content=payload)
    )
    dest = tmp_path / "adapter.bin"
    result = runner.invoke(app, ["training", "download", "run_d", str(dest)])
    assert result.exit_code == 0, result.stderr
    assert env_route.called
    assert blob_route.called
    assert dest.read_bytes() == payload
    # Regression: the dead /artifact route is never registered, so a request to
    # it would raise an unmocked-route error — the asserts above prove the CLI
    # only ever touched /download + the signed URL.


@respx.mock
def test_download_rejects_sha256_mismatch(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A corrupted transfer (sha mismatch vs the gateway-pinned digest) must
    fail and not leave a half/wrong file behind."""
    respx.get(f"{BASE}/training/runs/run_e/download").mock(
        return_value=httpx.Response(
            200,
            json={
                "url": "https://r2.example/signed/run_e.bin",
                "expires_in": 3600,
                "sha256": "0" * 64,  # wrong on purpose
                "size_bytes": 4,
            },
        )
    )
    respx.get("https://r2.example/signed/run_e.bin").mock(
        return_value=httpx.Response(200, content=b"data")
    )
    dest = tmp_path / "bad.bin"
    result = runner.invoke(app, ["training", "download", "run_e", str(dest)])
    assert result.exit_code != 0
    assert not dest.exists()
