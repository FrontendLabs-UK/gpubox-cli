"""Unit coverage for `gpb argus ...` (V1.5 W4).

Drives the Typer commands through CliRunner with a stubbed GPUBoxClient so no
network is touched. Asserts the path style (no /v1 prefix — base_url carries
it), the request verbs, the scope payload, the active-workspace header pinning,
the idempotency flags, and the inbox read flow.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.commands import argus as argus_cmd

runner = CliRunner()

_ACTIVE_WS = "ws-active-123"


class _StubClient:
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
            "params": params, "idempotent": idempotent,
            "extra_headers": extra_headers or {},
        })
        return self._responses.pop(0) if self._responses else {}


@contextmanager
def _patch_client(monkeypatch, responses):
    stub = _StubClient(responses)
    monkeypatch.setattr(argus_cmd, "_client", lambda ctx: stub)
    yield stub


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUBOX_CONFIG_DIR", str(tmp_path))
    # Pin an active workspace so we can assert the X-GPUBox-Workspace header.
    settings = cfg.load_settings()
    settings.extra["active_workspace"] = _ACTIVE_WS
    cfg.save_settings(settings)
    yield


def test_create_posts_to_argus_agents_without_v1_prefix(monkeypatch):
    with _patch_client(monkeypatch, [{"id": "a-1", "question": "rev?"}]) as stub:
        res = runner.invoke(argus_cmd.app, [
            "create", "-q", "What is revenue?", "-d", "doc-1", "-d", "doc-2",
            "--cadence", "daily"])
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/argus/agents"
    assert call["json_body"]["question"] == "What is revenue?"
    assert call["json_body"]["doc_scope_ids"] == ["doc-1", "doc-2"]
    # Tags are never sent (gateway rejects them in W4).
    assert call["json_body"]["doc_scope_tags"] == []
    assert call["json_body"]["cadence"] == "daily"
    # create is idempotent + carries the active-workspace header.
    assert call["idempotent"] is True
    assert call["extra_headers"]["X-GPUBox-Workspace"] == _ACTIVE_WS


def test_create_requires_at_least_one_doc(monkeypatch):
    with _patch_client(monkeypatch, [{}]):
        res = runner.invoke(argus_cmd.app, ["create", "-q", "no docs"])
    # Missing required --doc -> non-zero exit, no request made.
    assert res.exit_code != 0


def test_list_gets_agents(monkeypatch):
    with _patch_client(monkeypatch, [{"data": [
        {"id": "a-1", "cadence": "daily", "status": "active", "question": "q"}]}]) as stub:
        res = runner.invoke(argus_cmd.app, ["list"])
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "GET" and call["path"] == "/argus/agents"
    assert call["extra_headers"]["X-GPUBox-Workspace"] == _ACTIVE_WS
    assert "a-1" in res.output


def test_get_agent(monkeypatch):
    with _patch_client(monkeypatch, [{"id": "a-1", "cadence": "daily",
                                      "status": "active", "question": "q",
                                      "doc_scope_ids": ["doc-1"]}]) as stub:
        res = runner.invoke(argus_cmd.app, ["get", "a-1"])
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "GET" and call["path"] == "/argus/agents/a-1"


def test_run_agent_posts_to_run_path(monkeypatch):
    with _patch_client(monkeypatch, [{"run_id": "r-1", "status": "pending"}]) as stub:
        res = runner.invoke(argus_cmd.app, ["run", "a-1"])
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/argus/agents/a-1/run"
    # idempotent + active-workspace header pinned (mirrors create/read).
    assert call["idempotent"] is True
    assert call["extra_headers"]["X-GPUBox-Workspace"] == _ACTIVE_WS
    assert "r-1" in res.output


def test_run_agent_falls_back_to_id_field(monkeypatch):
    """The gateway returns {run_id, status}; tolerate a bare {id} too (so a CLI
    pinned against an older gateway that echoed `id` still prints the run id)."""
    with _patch_client(monkeypatch, [{"id": "r-legacy", "status": "pending"}]) as stub:
        res = runner.invoke(argus_cmd.app, ["run", "a-2"])
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["path"] == "/argus/agents/a-2/run"
    assert "r-legacy" in res.output


def test_delete_agent(monkeypatch):
    with _patch_client(monkeypatch, [{"id": "a-1", "deleted": True}]) as stub:
        res = runner.invoke(argus_cmd.app, ["delete", "a-1"])
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "DELETE" and call["path"] == "/argus/agents/a-1"


def test_inbox_read(monkeypatch):
    with _patch_client(monkeypatch, [{"data": [
        {"id": "it-1", "title": "Revenue", "body": "4.2m", "confidence": "high",
         "read": False}]}]) as stub:
        res = runner.invoke(argus_cmd.app, ["inbox", "--unread"])
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "GET" and call["path"] == "/argus/inbox"
    assert call["params"]["unread"] == "true"
    assert call["extra_headers"]["X-GPUBox-Workspace"] == _ACTIVE_WS
    assert "Revenue" in res.output


def test_mark_read(monkeypatch):
    with _patch_client(monkeypatch, [{"id": "it-1", "read": True}]) as stub:
        res = runner.invoke(argus_cmd.app, ["read", "it-1"])
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "POST" and call["path"] == "/argus/inbox/it-1/read"
    assert call["idempotent"] is True
