"""Unit coverage for `gpb finetune ...` (V1.5 W3).

Drives the Typer commands through CliRunner with a stubbed GPUBoxClient so no
network is touched. Asserts the path style (no /v1 prefix — base_url carries
it), the request verbs, that the active-workspace pin is sent as the
X-GPUBox-Workspace header, and the `use` pin/clear/show shapes.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.commands import finetune as ft_cmd

runner = CliRunner()


class _StubClient:
    """Records (method, path, json_body, params, extra_headers) and returns
    canned responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def request(self, method, path, *, json_body=None, params=None,
                idempotent=False, extra_headers=None):
        self.calls.append({
            "method": method, "path": path, "json_body": json_body,
            "params": params, "extra_headers": extra_headers,
        })
        return self._responses.pop(0) if self._responses else {}


@contextmanager
def _patch_client(monkeypatch, responses):
    stub = _StubClient(responses)
    monkeypatch.setattr(ft_cmd, "_client", lambda ctx: stub)
    yield stub


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUBOX_CONFIG_DIR", str(tmp_path))
    yield


def _pin_workspace(ws: str) -> None:
    settings = cfg.load_settings()
    settings.extra["active_workspace"] = ws
    cfg.save_settings(settings)


# ---------------------------------------------------------------------------
# create — POST /training/runs with workspace header
# ---------------------------------------------------------------------------


def test_create_posts_training_run_without_v1_prefix(monkeypatch):
    with _patch_client(monkeypatch, [{"id": "run-1"}]) as stub:
        res = runner.invoke(
            ft_cmd.app,
            ["create", "--preset", "qwen32b-lora-r16", "--dataset", "gpubox://ds"],
        )
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/training/runs"  # NO /v1 prefix
    assert call["json_body"] == {"preset": "qwen32b-lora-r16", "dataset": "gpubox://ds"}


def test_create_sends_active_workspace_header(monkeypatch):
    _pin_workspace("ws-A")
    with _patch_client(monkeypatch, [{"id": "run-1"}]) as stub:
        res = runner.invoke(
            ft_cmd.app,
            ["create", "--preset", "p", "--dataset", "d"],
        )
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["extra_headers"] == {"X-GPUBox-Workspace": "ws-A"}


def test_create_workspace_override_wins_over_pin(monkeypatch):
    _pin_workspace("ws-A")
    with _patch_client(monkeypatch, [{"id": "run-1"}]) as stub:
        res = runner.invoke(
            ft_cmd.app,
            ["create", "--preset", "p", "--dataset", "d", "--workspace", "ws-B"],
        )
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["extra_headers"] == {"X-GPUBox-Workspace": "ws-B"}


# ---------------------------------------------------------------------------
# list — runs (default) and adapters
# ---------------------------------------------------------------------------


def test_list_runs_default(monkeypatch):
    resp = {"items": [{"id": "run-1", "status": "running", "preset": "p"}]}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(ft_cmd.app, ["list"])
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["method"] == "GET"
    assert stub.calls[0]["path"] == "/training/runs"
    assert "run-1" in res.output


def test_list_adapters_hits_lora_endpoint(monkeypatch):
    resp = {"object": "list", "data": [
        {"name": "acme-v1", "version": 1, "status": "registered", "family": "qwen-32b"}
    ]}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(ft_cmd.app, ["list", "--adapters"])
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["path"] == "/lora/adapters"
    assert "acme-v1" in res.output


# ---------------------------------------------------------------------------
# status — GET /training/runs/{id}
# ---------------------------------------------------------------------------


def test_status_gets_run(monkeypatch):
    with _patch_client(monkeypatch, [{"id": "run-9", "status": "succeeded"}]) as stub:
        res = runner.invoke(ft_cmd.app, ["status", "run-9"])
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["method"] == "GET"
    assert stub.calls[0]["path"] == "/training/runs/run-9"
    assert "succeeded" in res.output


# ---------------------------------------------------------------------------
# use — PUT/GET/DELETE /finetune/active
# ---------------------------------------------------------------------------


def test_use_puts_active_finetune(monkeypatch):
    resp = {"workspace_id": "ws-A", "hosted_model_name": "acme-v1",
            "chat_model": "lora:acme-v1"}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(ft_cmd.app, ["use", "acme-v1"])
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "PUT"
    assert call["path"] == "/finetune/active"
    assert call["json_body"] == {"hosted_model_name": "acme-v1"}
    assert "lora:acme-v1" in res.output


def test_use_clear_deletes(monkeypatch):
    resp = {"workspace_id": "ws-A", "hosted_model_name": None, "chat_model": None}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(ft_cmd.app, ["use", "ignored", "--clear"])
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["method"] == "DELETE"
    assert stub.calls[0]["path"] == "/finetune/active"


def test_use_show_gets(monkeypatch):
    resp = {"workspace_id": "ws-A", "hosted_model_name": "acme-v1",
            "chat_model": "lora:acme-v1"}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(ft_cmd.app, ["use", "ignored", "--show"])
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["method"] == "GET"
    assert stub.calls[0]["path"] == "/finetune/active"


def test_use_sends_workspace_header(monkeypatch):
    _pin_workspace("ws-Z")
    resp = {"workspace_id": "ws-Z", "hosted_model_name": "m", "chat_model": "lora:m"}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(ft_cmd.app, ["use", "m"])
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["extra_headers"] == {"X-GPUBox-Workspace": "ws-Z"}
