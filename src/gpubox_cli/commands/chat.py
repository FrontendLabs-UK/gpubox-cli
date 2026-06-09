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

import json
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
    messages: list[dict[str, str]],
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
    save_session: Path | None = typer.Option(
        None,
        "--save-session",
        help="Append the conversation transcript to a JSON-lines file (opt-in).",
    ),
) -> None:
    """Send a chat completion. Streams to TTY, plain-prints when piped."""
    ctx_obj = ctx.obj or {}
    out: OutputCtx = ctx_obj.get("output", OutputCtx())
    chosen_model = _resolve_model(ctx_obj, model)

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

    if not prompt:
        emit_error(out, "missing prompt. example: gpb chat \"hello world\"")
        raise typer.Exit(2)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        with _build_client(ctx_obj) as client:
            assistant = _stream_chat(client, messages, chosen_model, out, extra)
    except GPUBoxError:
        # main._entrypoint will format. Re-raise.
        raise

    if save_session:
        _append_session(save_session, messages, assistant, chosen_model)


def _append_session(path: Path, messages: list[dict[str, str]], assistant: str, model: str) -> None:
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

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

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
                done = _handle_slash(user_input, messages, out, save_session)
                if done == "model":
                    # /model toggles the local var
                    model = _last_model_change[0] or model
                if done == "exit":
                    break
                continue

            messages.append({"role": "user", "content": user_input})
            try:
                assistant = _stream_chat(client, messages, model, out, extra)
            except GPUBoxError as exc:
                emit_error(out, str(exc))
                # Roll back the unanswered user turn so the next try is clean.
                messages.pop()
                continue
            messages.append({"role": "assistant", "content": assistant})

            if save_session:
                _append_session(save_session, messages[:-1], assistant, model)


# Tiny shared cell so /model can mutate the outer scope's `model` variable.
_last_model_change: list[str | None] = [None]


def _handle_slash(
    line: str,
    messages: list[dict[str, str]],
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
