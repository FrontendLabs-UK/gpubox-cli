"""`gpb training ...` — submit / list / watch / download / cancel runs.

Wave 5/8/9 backend. The CLI's verbs match what the Factory API exposes.
``watch`` polls every few seconds until the run reaches a terminal state.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_progress, emit_text
from gpubox_cli.version import USER_AGENT

app = typer.Typer(no_args_is_help=True, help="Submit + monitor fine-tuning runs.")
_TERMINAL_STATES = {"succeeded", "failed", "cancelled", "completed"}


def _output(ctx: typer.Context) -> OutputCtx:
    return (ctx.obj or {}).get("output", OutputCtx())


def _client(ctx: typer.Context) -> GPUBoxClient:
    obj = ctx.obj or {}
    resolved = cfg.resolve(
        profile_override=obj.get("profile"),
        api_key_override=obj.get("api_key"),
        base_url_override=obj.get("base_url"),
    )
    return GPUBoxClient(ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url))


@app.command("submit")
@exit_on_error
def submit(
    ctx: typer.Context,
    preset: str = typer.Option(..., "--preset", help="Factory preset (e.g. deberta-base)."),
    dataset: str = typer.Option(
        ..., "--dataset", help="Dataset URI (s3://, https://, gpubox://...)."
    ),
    name: str | None = typer.Option(None, "--name", help="Optional run label."),
    epochs: int | None = typer.Option(None, "--epochs", min=1),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1),
    learning_rate: float | None = typer.Option(None, "--learning-rate"),
) -> None:
    """Submit a new training run. Returns the run id."""
    out = _output(ctx)
    body: dict = {"preset": preset, "dataset": dataset}
    if name:
        body["name"] = name
    if epochs:
        body["epochs"] = epochs
    if batch_size:
        body["batch_size"] = batch_size
    if learning_rate is not None:
        body["learning_rate"] = learning_rate

    with _client(ctx) as client:
        resp = client.request("POST", "/training/runs", json_body=body, idempotent=True)
    if out.json_mode:
        emit_json(out, resp)
        return
    rid = resp.get("id") if isinstance(resp, dict) else None
    emit_text(out, f"submitted run: {rid}" if rid else str(resp))


@app.command("list")
@exit_on_error
def list_runs(
    ctx: typer.Context,
    status: str | None = typer.Option(None, "--status", help="Filter by status."),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """List training runs, optionally filtered by status."""
    out = _output(ctx)
    params: dict = {"limit": limit}
    if status:
        params["status"] = status
    with _client(ctx) as client:
        resp = client.request("GET", "/training/runs", params=params)
    if out.json_mode:
        emit_json(out, resp)
        return
    items = resp.get("items", []) if isinstance(resp, dict) else []
    if not items:
        emit_text(out, "(no runs)")
        return
    for item in items:
        emit_text(
            out,
            f"{item.get('id','?'):<24} {item.get('status','?'):<12} "
            f"{item.get('preset','?'):<24} {item.get('created_at','?')}",
        )


@app.command("status")
@exit_on_error
def status_run(ctx: typer.Context, run_id: str = typer.Argument(...)) -> None:
    """One-shot status of a run."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", f"/training/runs/{run_id}")
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"id:       {resp.get('id')}")
    emit_text(out, f"status:   {resp.get('status')}")
    if "progress" in resp:
        emit_text(out, f"progress: {resp.get('progress')}")
    if "metrics" in resp:
        emit_text(out, f"metrics:  {resp.get('metrics')}")


@app.command("watch")
@exit_on_error
def watch_run(
    ctx: typer.Context,
    run_id: str = typer.Argument(...),
    interval: float = typer.Option(5.0, "--interval", min=1.0, max=60.0),
) -> None:
    """Poll a run until it reaches a terminal state."""
    out = _output(ctx)
    with _client(ctx) as client:
        while True:
            resp = client.request("GET", f"/training/runs/{run_id}")
            state = (resp.get("status") or "").lower() if isinstance(resp, dict) else ""
            line = f"{run_id} status={state} progress={resp.get('progress','-')}"
            # Progress goes through emit_progress so --json keeps stdout as
            # exactly one JSON document (progress reroutes to stderr there).
            emit_progress(out, line)
            if state in _TERMINAL_STATES:
                if out.json_mode:
                    emit_json(out, resp)
                if state in ("failed", "cancelled"):
                    raise typer.Exit(1)
                return
            time.sleep(interval)


@app.command("download")
@exit_on_error
def download_run(
    ctx: typer.Context,
    run_id: str = typer.Argument(...),
    output_path: Path = typer.Argument(...),
) -> None:
    """Download the model artefact produced by a successful run."""
    out = _output(ctx)
    obj = ctx.obj or {}
    resolved = cfg.resolve(
        profile_override=obj.get("profile"),
        api_key_override=obj.get("api_key"),
        base_url_override=obj.get("base_url"),
    )
    if not resolved.api_key:
        emit_error(out, "no API key configured")
        raise typer.Exit(4)

    url = resolved.base_url.rstrip("/") + f"/training/runs/{run_id}/artifact"
    headers = {
        "Authorization": f"Bearer {resolved.api_key}",
        "User-Agent": USER_AGENT,
        "Accept": "application/octet-stream",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream("GET", url, headers=headers, timeout=600.0, follow_redirects=True) as resp:
            if resp.status_code >= 400:
                # Route through the typed-error helper for consistent UX:
                # 401 → AuthError exit 4, 402 → PaymentRequiredError exit 5, etc.
                with GPUBoxClient(
                    ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url)
                ) as client:
                    resp.read()
                    client.raise_for_response(resp)
            with output_path.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 16):
                    fh.write(chunk)
    except httpx.HTTPError as exc:
        from gpubox_cli.client import GPUBoxError

        raise GPUBoxError(f"network error during download: {exc}") from exc

    if not out.quiet:
        emit_text(out, f"saved to {output_path}", end="\n")


@app.command("cancel")
@exit_on_error
def cancel_run(ctx: typer.Context, run_id: str = typer.Argument(...)) -> None:
    """Cancel a running training job."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("POST", f"/training/runs/{run_id}/cancel")
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"cancelled: {run_id}")
