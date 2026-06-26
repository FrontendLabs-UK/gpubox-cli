"""Unit coverage for `gpb assistants ...` (Wave 7.4 custom assistants).

Asserts the corrected gateway contract after the CLI<->gateway drift fix:
  * create  → POST /assistants with REQUIRED `slug` (+ name, instructions[, model])
  * update  → POST /assistants/{id} (NOT PATCH — there is no PATCH handler)
  * run     → POST /chat/completions with model alias `asst_<id>`
              (NOT /assistants/{id}/runs, which does not exist)
Plus regression asserts that the previously-sent shapes are gone.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.commands import assistants as asst_cmd

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("GPB_CONFIG_DIR", str(tmp_path))
    yield


def _pin_workspace(ws: str) -> None:
    settings = cfg.load_settings()
    settings.extra["active_workspace"] = ws
    cfg.save_settings(settings)


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
            "extra_headers": extra_headers,
        })
        return self._responses.pop(0) if self._responses else {}


@contextmanager
def _patch_client(monkeypatch, responses):
    stub = _StubClient(responses)
    monkeypatch.setattr(asst_cmd, "_client", lambda ctx: stub)
    yield stub


# ---------------------------------------------------------------------------
# create — POST /assistants with REQUIRED slug
# ---------------------------------------------------------------------------


def test_create_sends_slug_name_instructions(monkeypatch, tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("you are helpful", encoding="utf-8")
    with _patch_client(monkeypatch, [{"id": "asst_1"}]) as stub:
        res = runner.invoke(
            asst_cmd.app,
            ["create", "--slug", "support-bot", "--name", "Support",
             "--instructions", str(prompt)],
        )
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/assistants"
    assert call["idempotent"] is True
    assert call["json_body"] == {
        "slug": "support-bot",
        "name": "Support",
        "instructions": "you are helpful",
    }


def test_create_includes_model_when_given(monkeypatch, tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("hi", encoding="utf-8")
    with _patch_client(monkeypatch, [{"id": "asst_1"}]) as stub:
        res = runner.invoke(
            asst_cmd.app,
            ["create", "--slug", "s-bot", "--name", "S",
             "--instructions", str(prompt), "--model", "qwen2.5-32b-instruct"],
        )
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["json_body"]["model"] == "qwen2.5-32b-instruct"


def test_create_requires_slug(monkeypatch, tmp_path):
    """slug is REQUIRED by CreateAssistantRequest — omitting it must fail at the
    CLI, not 422 on body.slug at the gateway."""
    prompt = tmp_path / "p.txt"
    prompt.write_text("hi", encoding="utf-8")
    with _patch_client(monkeypatch, [{"id": "asst_1"}]):
        res = runner.invoke(
            asst_cmd.app,
            ["create", "--name", "S", "--instructions", str(prompt)],
        )
    assert res.exit_code != 0


# ---------------------------------------------------------------------------
# update — POST /assistants/{id}  (no PATCH handler exists)
# ---------------------------------------------------------------------------


def test_update_uses_post_not_patch(monkeypatch, tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("updated", encoding="utf-8")
    with _patch_client(monkeypatch, [{"id": "asst_9"}]) as stub:
        res = runner.invoke(
            asst_cmd.app,
            ["update", "asst_9", "--name", "New", "--instructions", str(prompt)],
        )
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "POST"  # regression: was PATCH (405)
    assert call["method"] != "PATCH"
    assert call["path"] == "/assistants/asst_9"
    assert call["json_body"] == {"instructions": "updated", "name": "New"}


def test_update_empty_body_errors(monkeypatch):
    with _patch_client(monkeypatch, []):
        res = runner.invoke(asst_cmd.app, ["update", "asst_9"])
    assert res.exit_code == 2


# ---------------------------------------------------------------------------
# run — POST /chat/completions with model=asst_<id>
# ---------------------------------------------------------------------------


def test_run_routes_through_chat_completions(monkeypatch):
    resp = {"choices": [{"message": {"role": "assistant", "content": "hello back"}}]}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(asst_cmd.app, ["run", "asst_42", "say hi"])
    assert res.exit_code == 0, res.output
    call = stub.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/chat/completions"  # regression: was /assistants/{id}/runs (404)
    assert call["json_body"] == {
        "model": "asst_42",
        "messages": [{"role": "user", "content": "say hi"}],
    }
    assert "hello back" in res.output


def test_run_prefixes_bare_uuid(monkeypatch):
    """A bare id (no asst_ prefix) is wrapped into the alias the gateway expects."""
    resp = {"choices": [{"message": {"content": "ok"}}]}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(asst_cmd.app, ["run", "abc-123", "hi"])
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["json_body"]["model"] == "asst_abc-123"


def test_run_never_hits_runs_subresource(monkeypatch):
    """Regression: the non-existent /runs sub-resource and {'input': ...} shape
    must be gone."""
    resp = {"choices": [{"message": {"content": "ok"}}]}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(asst_cmd.app, ["run", "asst_7", "go"])
    assert res.exit_code == 0, res.output
    body = stub.calls[0]["json_body"]
    assert not stub.calls[0]["path"].endswith("/runs")
    assert "input" not in body


# ---------------------------------------------------------------------------
# workspace header — assistants resolve through X-GPUBox-Workspace (V1.5 W1)
# ---------------------------------------------------------------------------


def test_create_sends_active_workspace_header(monkeypatch, tmp_path):
    """Without the workspace header the gateway resolves to Default; a fine-tune
    flow built in workspace A would then be invisible to the assistant."""
    _pin_workspace("ws-A")
    prompt = tmp_path / "p.txt"
    prompt.write_text("hi", encoding="utf-8")
    with _patch_client(monkeypatch, [{"id": "asst_1"}]) as stub:
        res = runner.invoke(
            asst_cmd.app,
            ["create", "--slug", "s-bot", "--name", "S", "--instructions", str(prompt)],
        )
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["extra_headers"] == {"X-GPUBox-Workspace": "ws-A"}


def test_run_workspace_override_wins_over_pin(monkeypatch):
    _pin_workspace("ws-A")
    resp = {"choices": [{"message": {"content": "ok"}}]}
    with _patch_client(monkeypatch, [resp]) as stub:
        res = runner.invoke(
            asst_cmd.app, ["run", "asst_1", "hi", "--workspace", "ws-B"]
        )
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["extra_headers"] == {"X-GPUBox-Workspace": "ws-B"}


def test_update_sends_workspace_header(monkeypatch, tmp_path):
    _pin_workspace("ws-Z")
    prompt = tmp_path / "p.txt"
    prompt.write_text("x", encoding="utf-8")
    with _patch_client(monkeypatch, [{"id": "asst_9"}]) as stub:
        res = runner.invoke(
            asst_cmd.app, ["update", "asst_9", "--instructions", str(prompt)]
        )
    assert res.exit_code == 0, res.output
    assert stub.calls[0]["extra_headers"] == {"X-GPUBox-Workspace": "ws-Z"}
