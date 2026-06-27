"""`gpubox audio speech` — text-to-speech via POST /v1/audio/speech.

The canonical OpenAI-compatible TTS surface (gateway PR feat/tts-canonical-speech).
The gateway owns /v1/audio/speech and proxies to a TTS upstream (CosyVoice/XTTS).

UX choice (mirrors `transcribe`): the response is raw AUDIO BYTES, so we
write them to `--output <file>`, or to stdout when stdout is not a TTY
(so `gpubox audio speech ... > out.mp3` and pipes Just Work). `--format`
maps to the gateway's `response_format` (mp3|opus|wav|pcm|mulaw_8000).

DARK on the gateway: while GPUBOX_TTS_ENABLED is OFF the gateway returns
503 `tts_disabled`; the CLI surfaces that error verbatim. The streaming
WebSocket surface (/v1/audio/speech/stream) is the live telephony hot
path and is NOT exposed here — the CLI is request/response only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient
from gpubox_cli.output import OutputCtx, emit_error, emit_progress
from gpubox_cli.version import USER_AGENT

app = typer.Typer(help="Text-to-speech and audio surfaces.", no_args_is_help=True)

DEFAULT_MODEL = "cosyvoice2"
DEFAULT_VOICE = "olamide"
DEFAULT_FORMAT = "mp3"

# response_format -> a sensible default file extension for --output omission.
_FORMAT_EXT = {
    "mp3": "mp3",
    "opus": "ogg",
    "wav": "wav",
    "pcm": "pcm",
    "mulaw_8000": "ulaw",
}


@app.command("speech")
def speech(
    ctx: typer.Context,
    text: str = typer.Argument(..., help="Text to synthesise (max 4000 chars)."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="TTS model id."),
    voice: str = typer.Option(DEFAULT_VOICE, "--voice", help="Voice name."),
    response_format: str = typer.Option(
        DEFAULT_FORMAT, "--format", "-f",
        help="Audio format: mp3|opus|wav|pcm|mulaw_8000.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Write audio to this file. Omitted: stdout if piped, else <speech>.<ext>.",
    ),
) -> None:
    """Synthesise speech from text and save the audio.

    One-shot httpx.post (not GPUBoxClient.request) because the response is
    binary audio, not JSON — but auth header + UA + base URL still come
    from the resolved config, identical to `transcribe`.
    """
    ctx_obj = ctx.obj or {}
    out: OutputCtx = ctx_obj.get("output", OutputCtx())
    resolved = cfg.resolve(
        profile_override=ctx_obj.get("profile"),
        api_key_override=ctx_obj.get("api_key"),
        base_url_override=ctx_obj.get("base_url"),
    )
    if not resolved.api_key:
        emit_error(out, "no API key configured. run `gpubox auth login`.")
        raise typer.Exit(4)

    headers = {
        "Authorization": f"Bearer {resolved.api_key}",
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
    }
    url = resolved.base_url.rstrip("/") + "/audio/speech"

    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=180.0)
    except httpx.HTTPError as exc:
        from gpubox_cli.client import GPUBoxError

        raise GPUBoxError(f"network error: {exc}") from exc

    if resp.status_code >= 400:
        # Reuse the public error mapper so the gateway's typed envelope
        # (tts_disabled, invalid_speed, invalid_response_format, ...) is
        # surfaced consistently.
        with GPUBoxClient(
            ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
        ) as client:
            client.raise_for_response(resp)  # raises GPUBoxError
        return

    audio = resp.content

    # Destination: explicit --output, else stdout when piped, else a file
    # named by the format extension.
    if output is not None:
        output.write_bytes(audio)
        emit_progress(out, f"wrote {len(audio)} bytes to {output}")
        return
    if not sys.stdout.isatty():
        sys.stdout.buffer.write(audio)
        sys.stdout.buffer.flush()
        return
    ext = _FORMAT_EXT.get(response_format, "bin")
    dest = Path(f"speech.{ext}")
    dest.write_bytes(audio)
    emit_progress(out, f"wrote {len(audio)} bytes to {dest}")
