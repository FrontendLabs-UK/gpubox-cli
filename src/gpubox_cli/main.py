"""Top-level Typer app + global flags.

The round-table converged on `gpb` as primary and `gpubox` as alias —
both names are wired in pyproject.toml's ``[project.scripts]`` and route
to this same ``app``.

Global flags (apply to every subcommand):

* ``--profile, -p``        named credential bundle (overrides GPB_PROFILE)
* ``--api-key``            one-shot key override (avoids touching files)
* ``--base-url``           one-shot endpoint override (staging, on-prem)
* ``--json, -j``           machine-readable output to stdout
* ``--quiet, -q``          suppress non-error output
* ``--verbose, -v``        debug/trace
* ``--no-color``           strip color (also honours NO_COLOR env)

The ``--api-key`` and ``--base-url`` overrides exist so a CI step can do
``gpb chat --api-key "$SECRET" "ping"`` without writing to disk.
"""

from __future__ import annotations

import sys

import typer

from gpubox_cli.client import GPUBoxError, render_error
from gpubox_cli.commands import (
    argus,
    assistants,
    audio,
    billing,
    chat,
    config_cmd,
    embed,
    finetune,
    hosting,
    profile_cmd,
    search,
    training,
    transcribe,
    users,
    vault,
    workspace,
)
from gpubox_cli.commands import auth as auth_cmd
from gpubox_cli.output import OutputCtx
from gpubox_cli.version import __version__

app = typer.Typer(
    name="gpb",
    help="GPUBox CLI — UK-sovereign AI inference. https://gpubox.ai",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"gpb {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    profile: str | None = typer.Option(
        None, "--profile", "-p", help="Named credential profile to use."
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key override (avoids touching credentials file).",
        envvar="GPUBOX_API_KEY",
        show_envvar=False,  # don't echo secret env name in --help noise
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Override API base URL (e.g. https://staging.gpubox.ai/v1).",
        envvar="GPUBOX_API_URL",
    ),
    json_out: bool = typer.Option(False, "--json", "-j", help="Emit JSON to stdout."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress info output."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug/trace output."),
    no_color: bool = typer.Option(
        False, "--no-color", help="Strip colour codes (also via NO_COLOR env)."
    ),
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show CLI version and exit.",
    ),
) -> None:
    """Stash global flags on ctx.obj for downstream commands.

    We keep this dict-shaped (not a custom class) so subcommands can read
    keys defensively; nothing here should crash when a key is missing.
    """
    ctx.obj = {
        "profile": profile,
        "api_key": api_key,
        "base_url": base_url,
        "output": OutputCtx(
            json_mode=json_out, quiet=quiet, verbose=verbose, no_color=no_color
        ),
    }


# ---------------------------------------------------------------------------
# Subcommand registration
# Each module exposes either a Typer sub-app or a callable; we mount them
# under stable noun names so users build muscle memory: gpb <noun> <verb>.
# ---------------------------------------------------------------------------

app.add_typer(auth_cmd.app, name="auth", help="Authentication and identity.")
app.add_typer(profile_cmd.app, name="profile", help="Manage credential profiles.")
app.add_typer(config_cmd.app, name="config", help="Read/write CLI config.")
app.add_typer(billing.app, name="billing", help="Wallet, top-ups, usage.")
app.add_typer(training.app, name="training", help="Submit + watch fine-tuning runs.")
app.add_typer(finetune.app, name="finetune", help="Workspace-scoped user fine-tunes (V1.5 W3).")
app.add_typer(hosting.app, name="hosting", help="Hosting tier promotion + management.")
app.add_typer(vault.app, name="vault", help="Conversation Vault + RAG corpora.")
app.command("search", help="Unified search + grounded synthesis (docs + chat).")(search.run)
app.add_typer(workspace.app, name="workspace", help="Workspaces — per-tenant isolation containers.")
app.add_typer(argus.app, name="argus", help="Argus — your Standing Research Agent (V1.5 W4).")
app.add_typer(assistants.app, name="assistants", help="Custom assistants.")
app.add_typer(audio.app, name="audio", help="Text-to-speech (audio speech).")
app.add_typer(users.app, name="users", help="Users, invites, OIDC clients.")

# Top-level inference commands — these are flat (no sub-noun) because they
# are the most common surface; users muscle-memory `gpb chat "..."`.
app.command("chat", help="One-shot chat completion or interactive REPL.")(chat.run)
app.command("embed", help="Embed text and emit a vector.")(embed.run)
app.command("transcribe", help="Transcribe an audio file via Whisper.")(transcribe.run)
app.command("signup", help="Open the signup page in a browser.")(auth_cmd.signup_command)


def _entrypoint() -> None:
    """CLI entry point with belt-and-braces GPUBoxError handling.

    Note: every command that talks to the API is decorated with
    @exit_on_error, which converts GPUBoxError → typer.Exit(code) inside
    the command boundary. That's the load-bearing path. This wrapper is a
    safety net for any code path that bypasses the decorator (e.g. errors
    raised in callbacks before the command body runs).
    """
    try:
        app()
    except GPUBoxError as exc:
        sys.exit(render_error(exc))


# pyproject.toml's [project.scripts] points `gpb` and `gpubox` at THIS
# function (gpubox_cli.main:main_entry) so installed binaries get the
# wrapper. The bare `app` name remains importable for tests.
def main_entry() -> None:  # pragma: no cover
    _entrypoint()


if __name__ == "__main__":  # pragma: no cover
    _entrypoint()
