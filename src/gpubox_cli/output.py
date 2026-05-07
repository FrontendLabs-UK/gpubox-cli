"""Formatted output helpers — TTY-aware streaming and JSON emission.

Round-table lock #6: TTY ⇒ rich/streaming, non-TTY ⇒ plain stdout, with
``--json`` always emitting machine-readable JSON. Diagnostics go to stderr;
stdout is reserved for the actual command result so users can pipe.

Round-table lock (Codex sharpening): "no color/spinners/streaming in
non-TTY, and ``--json`` for structured output." JSON-only-on-flag, never
silent JSON-because-piped — that would surprise users running ``gpb chat
"hi" | less`` who expect plain text.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from rich.console import Console


@dataclass
class OutputCtx:
    """Carries output preferences down the call stack.

    Built once in main.py from the global flags, then attached to the typer
    Context. Commands consult it instead of re-reading argv.
    """

    json_mode: bool = False
    quiet: bool = False
    verbose: bool = False
    no_color: bool = False

    @property
    def stdout_is_tty(self) -> bool:
        try:
            return sys.stdout.isatty()
        except (AttributeError, ValueError):
            return False

    @property
    def use_color(self) -> bool:
        if self.no_color or os.environ.get("NO_COLOR"):
            return False
        return self.stdout_is_tty and not self.json_mode

    @property
    def use_streaming_render(self) -> bool:
        """True when we should stream tokens live to stdout.

        Decoupled from ``use_color`` per Codex review: ``--no-color`` means
        "no ANSI escapes" — it must NOT also disable streaming, which is a
        separate UX axis. Plain-text streaming is still streaming.
        """
        return self.stdout_is_tty and not self.json_mode and not self.quiet


def make_console(ctx: OutputCtx, *, stderr: bool = False) -> Console:
    """Build a rich Console honouring our output preferences.

    stderr console honours --no-color too (Codex review: stderr should not
    secretly emit color when the user asked for plain output).
    """
    no_color_env = bool(os.environ.get("NO_COLOR"))
    if stderr:
        return Console(
            stderr=True, force_terminal=False, no_color=ctx.no_color or no_color_env
        )
    return Console(no_color=not ctx.use_color, force_terminal=ctx.stdout_is_tty)


def emit_json(ctx: OutputCtx, payload: Any) -> None:
    """Write a JSON document to stdout. Compact-but-readable."""
    sys.stdout.write(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_text(ctx: OutputCtx, text: str, *, end: str = "\n") -> None:
    """Plain stdout text — used when not in JSON mode."""
    if ctx.quiet:
        return
    sys.stdout.write(text)
    sys.stdout.write(end)
    sys.stdout.flush()


def emit_error(ctx: OutputCtx, message: str) -> None:
    """Diagnostics on stderr so stdout stays pipeable.

    Round-table lock #5 (zero-balance UX): when this is called for a 402,
    callers should follow up with the topup URL on a separate line — see
    client._handle_http_error.
    """
    console = make_console(ctx, stderr=True)
    console.print(f"[bold red]error:[/bold red] {message}")
