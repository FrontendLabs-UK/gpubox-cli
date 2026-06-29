"""`gpb transcribe ./audio.mp3` — multipart upload to /v1/audio/transcriptions.

Whisper-style API. We don't try to be clever about the audio format — we
hand the raw bytes to the gateway with the filename so it can sniff
content-type. Round-table didn't have a specific lock here; UX choice is
"print the transcript to stdout" so users can pipe it into `grep`.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

import httpx
import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text
from gpubox_cli.version import USER_AGENT

DEFAULT_MODEL = "whisper-large-v3"


@exit_on_error
def run(
    ctx: typer.Context,
    audio_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", "-m",
        help=(
            "Whisper model id. Default 'whisper-large-v3'. "
            "Selectable: 'whisper-large-v3-turbo', or 'ng-whisper-medium-v4b' "
            "(owned-model lane — routed to a dedicated upstream when wired on "
            "the gateway, otherwise falls through to the default Whisper)."
        ),
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="ISO-639-1 language hint (e.g. 'en', 'yo')."
    ),
    response_format: str = typer.Option(
        "json", "--format", help="Whisper response format (json, text, srt, vtt)."
    ),
) -> None:
    """Transcribe an audio file via Whisper.

    We use a one-shot httpx.post here instead of GPUBoxClient.request
    because multipart with file streaming is a different code path; the
    auth header + UA + base URL still come from the resolved config.
    """
    ctx_obj = ctx.obj or {}
    out: OutputCtx = ctx_obj.get("output", OutputCtx())
    resolved = cfg.resolve(
        profile_override=ctx_obj.get("profile"),
        api_key_override=ctx_obj.get("api_key"),
        base_url_override=ctx_obj.get("base_url"),
    )
    if not resolved.api_key:
        emit_error(out, "no API key configured. run `gpb auth login`.")
        raise typer.Exit(4)

    mime, _ = mimetypes.guess_type(audio_path.name)
    mime = mime or "application/octet-stream"
    headers = {
        "Authorization": f"Bearer {resolved.api_key}",
        "User-Agent": USER_AGENT,
    }
    data = {"model": model, "response_format": response_format}
    if language:
        data["language"] = language

    url = resolved.base_url.rstrip("/") + "/audio/transcriptions"
    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, mime)}
        try:
            resp = httpx.post(url, headers=headers, data=data, files=files, timeout=120.0)
        except httpx.HTTPError as exc:
            # exit_on_error wraps GPUBoxError; raise that for consistent UX.
            from gpubox_cli.client import GPUBoxError

            raise GPUBoxError(f"network error: {exc}") from exc

    if resp.status_code >= 400:
        # Use the public helper (no more private-API smell — Codex review).
        with GPUBoxClient(
            ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
        ) as client:
            client.raise_for_response(resp)  # raises GPUBoxError
        return

    if response_format == "json":
        payload = resp.json()
        if out.json_mode:
            emit_json(out, payload)
        else:
            emit_text(out, payload.get("text", ""))
    else:
        # text / srt / vtt — straight to stdout
        emit_text(out, resp.text, end="")
