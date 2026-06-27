"""`gpubox audio speech` — text-to-speech command."""

from __future__ import annotations

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
def test_speech_writes_audio_to_output(runner: CliRunner, tmp_path: Path) -> None:
    audio_bytes = b"ID3\x04\x00fake-mp3-bytes"
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(
            200, content=audio_bytes, headers={"content-type": "audio/mpeg"}
        )
    )
    dest = tmp_path / "out.mp3"
    result = runner.invoke(
        app,
        ["audio", "speech", "hello world", "--output", str(dest), "--format", "mp3"],
    )
    assert result.exit_code == 0, result.output
    assert dest.read_bytes() == audio_bytes


@respx.mock
def test_speech_sends_canonical_body(runner: CliRunner, tmp_path: Path) -> None:
    captured = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, content=b"\x00\x01")

    respx.post(f"{BASE}/audio/speech").mock(side_effect=_capture)
    dest = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["audio", "speech", "say this", "--voice", "olamide",
         "--format", "wav", "--output", str(dest)],
    )
    assert result.exit_code == 0, result.output
    assert captured["model"] == "cosyvoice2"
    assert captured["input"] == "say this"
    assert captured["voice"] == "olamide"
    assert captured["response_format"] == "wav"


@respx.mock
def test_speech_surfaces_gateway_error(runner: CliRunner, tmp_path: Path) -> None:
    # Gateway DARK: GPUBOX_TTS_ENABLED OFF -> 503 tts_disabled. The CLI
    # must exit non-zero and surface the typed error.
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(
            503,
            json={"error": {
                "message": "Text-to-speech is not enabled on this gateway.",
                "type": "service_unavailable", "code": "tts_disabled", "param": None,
            }},
        )
    )
    result = runner.invoke(
        app, ["audio", "speech", "hi", "--output", str(tmp_path / "x.mp3")]
    )
    assert result.exit_code != 0
