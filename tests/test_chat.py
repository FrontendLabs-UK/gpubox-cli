"""Chat command tests using respx for HTTP mocking."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app

BASE = cfg.DEFAULT_API_URL


@pytest.fixture
def authed_profile(fake_api_key: str) -> str:
    """Save a key into the default profile so commands have something to use."""
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key, base_url=BASE))
    return fake_api_key


@respx.mock
def test_chat_one_shot_buffered(runner: CliRunner, authed_profile: str) -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "cmpl_1",
                "choices": [{"message": {"role": "assistant", "content": "hello back"}}],
            },
        )
    )
    # --json forces buffered path so the SSE branch isn't required.
    result = runner.invoke(app, ["--json", "chat", "hi"])
    assert result.exit_code == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["choices"][0]["message"]["content"] == "hello back"


@respx.mock
def test_chat_402_renders_topup_url(runner: CliRunner, authed_profile: str) -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(402, json={"error": {"message": "wallet empty"}})
    )
    result = runner.invoke(app, ["--json", "chat", "hi"])
    # Round-table lock #5 — 402 must NOT pollute stdout with half-output,
    # AND must surface a topup URL on stderr.
    assert result.exit_code == 5
    assert "Top up" in result.stderr
    assert "gpubox.ai" in result.stderr


@respx.mock
def test_chat_401_returns_auth_error(runner: CliRunner, authed_profile: str) -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    result = runner.invoke(app, ["--json", "chat", "hi"])
    assert result.exit_code == 4
    assert "auth" in result.stderr.lower()


def test_chat_without_api_key_fails_fast(runner: CliRunner) -> None:
    # No profile saved, no env var, no flag → AuthError before any network call.
    result = runner.invoke(app, ["chat", "hi"])
    assert result.exit_code == 4
    assert "API key" in result.stderr or "api key" in result.stderr.lower()


@respx.mock
def test_chat_network_error_returns_clean_exit(
    runner: CliRunner, authed_profile: str
) -> None:
    """Per Codex review: ConnectError on stream open must NOT traceback —
    it should produce a typed GPUBoxError with exit code 1 and a hint."""
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ConnectError("dns"))
    result = runner.invoke(app, ["--json", "chat", "hi"])
    assert result.exit_code == 1
    assert "could not reach" in result.stderr.lower() or "network" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Vision / --image (multimodal) — added 2026-06-19 with the gateway VL model.
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_image_auto_routes_to_vision_model(
    runner: CliRunner, authed_profile: str, tmp_path
) -> None:
    """--image with no --model auto-switches to the VL model and sends
    OpenAI-compatible multimodal content (text part + inline data-URI)."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfakebytes")
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "a red square"}}]}
        )
    )
    result = runner.invoke(app, ["--json", "chat", "what is this?", "-I", str(img)])
    assert result.exit_code == 0, result.stderr
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "qwen2.5-vl-7b-instruct"
    content = body["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@respx.mock
def test_chat_image_respects_explicit_model(
    runner: CliRunner, authed_profile: str, tmp_path
) -> None:
    """An explicit --model is never overridden by the vision auto-route."""
    img = tmp_path / "shot.jpg"
    img.write_bytes(b"\xff\xd8\xfffake")
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    result = runner.invoke(
        app, ["--json", "chat", "hi", "-I", str(img), "-m", "my-custom-vl"]
    )
    assert result.exit_code == 0, result.stderr
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "my-custom-vl"


@respx.mock
def test_chat_image_url_passthrough(runner: CliRunner, authed_profile: str) -> None:
    """An https image URL is passed through verbatim (no base64 wrap)."""
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    url = "https://example.com/diagram.png"
    result = runner.invoke(app, ["--json", "chat", "describe", "-I", url])
    assert result.exit_code == 0, result.stderr
    body = json.loads(route.calls.last.request.content)
    assert body["messages"][-1]["content"][1]["image_url"]["url"] == url


@respx.mock
def test_chat_image_empty_prompt_defaults_text(
    runner: CliRunner, authed_profile: str, tmp_path
) -> None:
    """`gpb chat -I img` with no prompt sends a sensible default instruction
    (so an image-only invocation isn't rejected as a missing prompt)."""
    img = tmp_path / "x.png"
    img.write_bytes(b"bytes")
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    result = runner.invoke(app, ["--json", "chat", "-I", str(img)])
    assert result.exit_code == 0, result.stderr
    body = json.loads(route.calls.last.request.content)
    assert body["messages"][-1]["content"][0]["text"] == "Describe this image."


def test_chat_missing_image_errors(runner: CliRunner, authed_profile: str) -> None:
    """A missing image path fails clean (non-zero, no traceback, helpful msg)
    BEFORE any network call."""
    result = runner.invoke(app, ["chat", "hi", "-I", "/no/such/image.png"])
    assert result.exit_code != 0
    assert "image not found" in (result.stdout + result.stderr).lower()


def test_chat_image_size_cap(
    runner: CliRunner, authed_profile: str, tmp_path, monkeypatch
) -> None:
    """A local image over MAX_IMAGE_BYTES fails clean before any base64/network."""
    from gpubox_cli.commands import chat as chat_mod

    monkeypatch.setattr(chat_mod, "MAX_IMAGE_BYTES", 4)
    big = tmp_path / "big.png"
    big.write_bytes(b"this is more than four bytes")
    result = runner.invoke(app, ["chat", "hi", "-I", str(big)])
    assert result.exit_code != 0
    assert "too large" in (result.stdout + result.stderr).lower()


def test_image_content_part_and_vision_detect(tmp_path) -> None:
    """Unit cover for the image helper + vision-model heuristic."""
    import pytest as _pytest

    from gpubox_cli.client import GPUBoxError
    from gpubox_cli.commands.chat import _image_content_part, _is_vision_model

    assert _image_content_part("https://x/y.png")["image_url"]["url"] == "https://x/y.png"
    assert _image_content_part("data:image/png;base64,AAA")["image_url"]["url"].startswith(
        "data:"
    )
    p = tmp_path / "a.png"
    p.write_bytes(b"hello")
    assert _image_content_part(str(p))["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    with _pytest.raises(GPUBoxError):
        _image_content_part("/no/file.png")

    assert _is_vision_model("qwen2.5-vl-7b-instruct")
    assert _is_vision_model("my-org/custom-vision-7b")
    assert not _is_vision_model("qwen2.5-32b-instruct")
