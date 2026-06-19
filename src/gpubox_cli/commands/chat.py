"""`gpb chat ...` — one-shot completion or interactive REPL.

Round-table locks honoured here:

* **#6 streaming**: tokens are streamed via SSE in a TTY; piped/--json
  collapses to a single buffered response. We use ``client.stream`` so
  4xx errors (especially 402) raise BEFORE the first byte hits stdout.
* **#7 REPL**: prompt_toolkit-based, multiline, slash-commands. Per
  Codex's privacy nudge, conversation persistence is OPT-IN
  (``--save-session <file>``) — we never silently dump prompts to disk.
* **OpenAI compat**: payloads match ``/v1/chat/completions`` shape so the
  same gateway endpoint backs the Python SDK and us.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, GPUBoxError, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text

#: Default model when neither --model nor a configured default is set.
#: Pinned to qwen2.5-32b-instruct per the V1 Chat direction lock
#: (2026-05-29): Qwen 2.5 32B is the default chat model on prod. The
#: prior gpubox/llama-3.1-8b-instruct slug was never live on the gateway
#: (gateway lists qwen2.5-32b-instruct + llama-3.3-70b-instruct as live)
#: so every CLI call without --model was hitting a 404.
FALLBACK_MODEL = "qwen2.5-32b-instruct"

#: Vision-language model on the gateway. When --image is passed and the
#: caller hasn't pinned a (vision) --model, we auto-route here so users
#: don't have to remember the slug. Same OpenAI-compatible
#: /v1/chat/completions endpoint — the request just carries image_url
#: content parts. Served on prod 2026-06-19.
VISION_MODEL = "qwen2.5-vl-7b-instruct"

#: CLI-side cap on a single local image before base64-inlining it. Guards
#: against accidentally sending a huge file (memory spike + a multi-MB JSON
#: body the gateway would reject anyway). URLs/data-URIs aren't read here so
#: they aren't capped by this.
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB


def _is_vision_model(model: str) -> bool:
    """Heuristic: does this model id look like a vision/multimodal model?

    Used to decide whether attaching --image should auto-switch the model.
    If the caller already pointed at a VL model (e.g. their own fine-tune),
    we leave it alone.
    """
    m = model.lower()
    return "-vl" in m or "vl-" in m or "vision" in m


def _image_content_part(ref: str) -> dict[str, Any]:
    """Build an OpenAI-compatible ``image_url`` content part.

    ``ref`` may be an http(s) URL, an existing ``data:`` URI, or a local
    file path. Local files are read and base64-encoded into a data URI so
    the image travels inline with the request (no separate upload, works
    against any OpenAI-compatible endpoint). Raises GPUBoxError on a
    missing/unreadable file so the @exit_on_error decorator renders a
    clean non-zero exit.
    """
    if ref.startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": ref}}

    path = Path(ref).expanduser()
    if not path.is_file():
        raise GPUBoxError(
            f"image not found: {ref}",
            hint="pass a readable file path, an https URL, or a data: URI",
        )
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise GPUBoxError(
            f"image too large: {ref} is {size // (1024 * 1024)} MiB "
            f"(limit {MAX_IMAGE_BYTES // (1024 * 1024)} MiB)",
            hint="resize/compress the image, or pass an https URL instead",
        )
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        # Default to PNG; the gateway/model sniffs the actual bytes anyway.
        mime = "image/png"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:  # pragma: no cover - unreadable file race
        raise GPUBoxError(f"could not read image {ref}: {exc}") from exc
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _resolve_model(
    out_ctx: dict, explicit: str | None
) -> str:
    """Pick the model in this priority: --model > profile default > FALLBACK."""
    if explicit:
        return explicit
    resolved = cfg.resolve(profile_override=out_ctx.get("profile"))
    return resolved.default_model or FALLBACK_MODEL


def _build_client(ctx_obj: dict) -> GPUBoxClient:
    resolved = cfg.resolve(
        profile_override=ctx_obj.get("profile"),
        api_key_override=ctx_obj.get("api_key"),
        base_url_override=ctx_obj.get("base_url"),
    )
    return GPUBoxClient(
        ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
    )


def _stream_chat(
    client: GPUBoxClient,
    messages: list[dict[str, Any]],
    model: str,
    out: OutputCtx,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stream a chat completion and return the assembled text.

    The actual "render to TTY" behaviour is split out: we write tokens to
    stdout immediately when streaming is on, and accumulate the full text
    for callers that need it (e.g. REPL appends assistant turn back into
    the messages array).
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": out.use_streaming_render,
    }
    if extra:
        body.update(extra)

    accumulated: list[str] = []

    if not out.use_streaming_render:
        # Buffered path — for --json, --quiet, or non-TTY.
        resp = client.request("POST", "/chat/completions", json_body=body)
        text = _extract_text(resp)
        accumulated.append(text)
        if out.json_mode:
            emit_json(out, resp)
        else:
            emit_text(out, text)
        return "".join(accumulated)

    # SSE streaming path — uses client.stream so 4xx fails before bytes.
    # Mid-stream transport errors (ReadError, RemoteProtocolError) on a flaky
    # link are caught and re-raised as GPUBoxError so the @exit_on_error
    # decorator on the command turns them into a clean non-zero exit.
    import httpx as _httpx

    with client.stream("POST", "/chat/completions", json_body=body) as response:
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        # Skip malformed SSE frames — gateway sometimes emits
                        # keepalive comments; don't crash the stream on those.
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        accumulated.append(delta)
                        sys.stdout.write(delta)
                        sys.stdout.flush()
        except _httpx.HTTPError as exc:
            from gpubox_cli.client import GPUBoxError as _GPUBoxError

            # Newline before the error so the partial output isn't on the
            # same line as the diagnostic; users see what they got + why.
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise _GPUBoxError(
                f"connection dropped mid-stream: {exc}",
                hint="re-run the command; partial output above was received",
            ) from exc
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(accumulated)


def _extract_text(resp: Any) -> str:
    """Pull the assistant message from a chat completion response."""
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


@exit_on_error
def run(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(
        None, help="Prompt text. Omit when using --interactive."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model id (defaults to configured default)."
    ),
    system: str | None = typer.Option(
        None, "--system", "-s", help="Optional system instruction."
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", min=0.0, max=2.0, help="Sampling temperature."
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", min=1, help="Max output tokens."
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Open a REPL instead of one-shot."
    ),
    image: list[str] | None = typer.Option(
        None,
        "--image",
        "-I",
        help=(
            "Attach an image (file path, https URL, or data: URI). Repeatable. "
            "Routes to the vision model unless --model is set."
        ),
    ),
    save_session: Path | None = typer.Option(
        None,
        "--save-session",
        help="Append the conversation transcript to a JSON-lines file (opt-in).",
    ),
) -> None:
    """Send a chat completion. Streams to TTY, plain-prints when piped.

    Pass --image one or more times to send images to the vision model
    (qwen2.5-vl-7b-instruct), e.g. ``gpb chat "what's wrong here?" -I bug.png``.
    """
    ctx_obj = ctx.obj or {}
    out: OutputCtx = ctx_obj.get("output", OutputCtx())
    chosen_model = _resolve_model(ctx_obj, model)

    # Vision auto-route: if images are attached and the caller didn't pin a
    # (vision) model, switch to the VL model so users don't need the slug.
    if image and not model and not _is_vision_model(chosen_model):
        chosen_model = VISION_MODEL

    extra: dict[str, Any] = {}
    if temperature is not None:
        extra["temperature"] = temperature
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens

    if interactive:
        _run_repl(
            ctx_obj, out, chosen_model, system=system, extra=extra, save_session=save_session
        )
        return

    if not prompt and not image:
        emit_error(out, "missing prompt. example: gpb chat \"hello world\"")
        raise typer.Exit(2)

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if image:
        # OpenAI-compatible multimodal content: a text part + one image_url
        # part per --image. Empty prompt with an image defaults to a sensible
        # ask so the model has an instruction.
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": prompt or "Describe this image."}
        ]
        parts.extend(_image_content_part(ref) for ref in image)
        messages.append({"role": "user", "content": parts})
    else:
        messages.append({"role": "user", "content": prompt})

    try:
        with _build_client(ctx_obj) as client:
            assistant = _stream_chat(client, messages, chosen_model, out, extra)
    except GPUBoxError:
        # main._entrypoint will format. Re-raise.
        raise

    if save_session:
        _append_session(save_session, messages, assistant, chosen_model)


def _append_session(path: Path, messages: list[dict[str, Any]], assistant: str, model: str) -> None:
    """JSONL append — one line per turn, append-only, plays nice with tail -f."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"model": model, "messages": messages + [{"role": "assistant", "content": assistant}]}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# REPL — prompt_toolkit, slash-commands, multi-turn memory.
# ---------------------------------------------------------------------------
_REPL_HELP = """\
slash commands:
  /clear        forget conversation history
  /system <txt> set / replace system message
  /model <id>   switch to another model for the next turn
  /image <ref>  attach an image (path/URL/data-URI) to your NEXT message
  /save <path>  append the current transcript to a JSONL file
  /exit         leave the REPL (Ctrl-D also works)
"""


def _run_repl(
    ctx_obj: dict,
    out: OutputCtx,
    model: str,
    *,
    system: str | None,
    extra: dict[str, Any],
    save_session: Path | None,
) -> None:
    """Multi-turn chat REPL with slash commands.

    History file lives in the config dir (NOT $HOME) so headless runs with
    GPB_CONFIG_DIR=/tmp Just Work. Conversation memory is in-process only
    until the user runs /save or passes --save-session — privacy default.
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
    except ImportError as exc:  # pragma: no cover
        emit_error(out, f"REPL requires prompt_toolkit: {exc}")
        raise typer.Exit(1) from exc

    history_dir = cfg.config_dir() / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    session = PromptSession(history=FileHistory(str(history_dir / "chat.history")))

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})

    # Images queued by /image, flushed into the next user turn. Stays empty
    # for text-only sessions so the common path is unchanged.
    pending_images: list[str] = []

    if not out.quiet:
        emit_text(out, f"GPUBox REPL — model={model}. /help for commands, Ctrl-D to exit.")

    with _build_client(ctx_obj) as client:
        while True:
            try:
                user_input = session.prompt("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                if not out.quiet:
                    emit_text(out, "\nbye.")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                # /image is handled here (not in _handle_slash) because it
                # mutates the loop-local pending_images / model.
                low = user_input.split(maxsplit=1)
                if low[0].lower() == "/image":
                    if len(low) < 2 or not low[1].strip():
                        emit_text(out, "usage: /image <path-or-url-or-data-uri>")
                    else:
                        pending_images.append(low[1].strip())
                        emit_text(out, f"attached image (sends with next message): {low[1].strip()}")
                    continue
                done = _handle_slash(user_input, messages, out, save_session)
                if done == "model":
                    # /model toggles the local var
                    model = _last_model_change[0] or model
                if done == "exit":
                    break
                continue

            # Build this turn. With pending images, send OpenAI-compatible
            # multimodal content and route to the vision model for this turn.
            turn_model = model
            turn_had_images = bool(pending_images)
            if pending_images:
                try:
                    parts: list[dict[str, Any]] = [{"type": "text", "text": user_input}]
                    parts.extend(_image_content_part(ref) for ref in pending_images)
                except GPUBoxError as exc:
                    emit_error(out, str(exc))
                    continue  # keep pending_images so the user can fix the path
                messages.append({"role": "user", "content": parts})
                if not _is_vision_model(turn_model):
                    turn_model = VISION_MODEL
            else:
                messages.append({"role": "user", "content": user_input})

            try:
                assistant = _stream_chat(client, messages, turn_model, out, extra)
            except GPUBoxError as exc:
                emit_error(out, str(exc))
                # Roll back the unanswered user turn so the next try is clean.
                # pending_images is left intact so a retry re-sends the images.
                messages.pop()
                continue
            messages.append({"role": "assistant", "content": assistant})
            # Images are consumed only after the turn actually succeeds.
            if turn_had_images:
                pending_images = []

            if save_session:
                _append_session(save_session, messages[:-1], assistant, turn_model)


# Tiny shared cell so /model can mutate the outer scope's `model` variable.
_last_model_change: list[str | None] = [None]


def _handle_slash(
    line: str,
    messages: list[dict[str, Any]],
    out: OutputCtx,
    save_session: Path | None,
) -> str | None:
    """Dispatch slash-commands; return a sentinel string for special handling."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        return "exit"
    if cmd in ("/help", "/?"):
        emit_text(out, _REPL_HELP)
        return None
    if cmd == "/clear":
        messages.clear()
        emit_text(out, "history cleared.")
        return None
    if cmd == "/system":
        # Replace the leading system message (or insert one).
        new = [m for m in messages if m["role"] != "system"]
        if arg:
            new.insert(0, {"role": "system", "content": arg})
        messages.clear()
        messages.extend(new)
        emit_text(out, f"system set to: {arg or '(none)'}")
        return None
    if cmd == "/model":
        if not arg:
            emit_text(out, "usage: /model <model-id>")
            return None
        _last_model_change[0] = arg.strip()
        emit_text(out, f"model -> {arg.strip()}")
        return "model"
    if cmd == "/save":
        if not arg:
            emit_text(out, "usage: /save <path>")
            return None
        path = Path(arg).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        emit_text(out, f"appended transcript to {path}")
        return None
    emit_text(out, f"unknown command: {cmd}. /help for the list.")
    return None
