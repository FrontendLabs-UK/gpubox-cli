"""`gpb training ...` — submit / list / watch / download / cancel runs.

Wave 5/8/9 backend. The CLI's verbs match what the Factory API exposes.
``watch`` polls every few seconds until the run reaches a terminal state.
"""

from __future__ import annotations

import hashlib
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


def _build_hyperparams(
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
) -> dict:
    """Collect per-run tunables into a `hyperparams` override dict.

    Key names mirror config/training.yaml's `default_hyperparams`
    (epochs / batch_size / learning_rate); the gateway shallow-merges this over
    the preset defaults. Returns {} when nothing was supplied.
    """
    hyperparams: dict = {}
    if epochs is not None:
        hyperparams["epochs"] = epochs
    if batch_size is not None:
        hyperparams["batch_size"] = batch_size
    if learning_rate is not None:
        hyperparams["learning_rate"] = learning_rate
    return hyperparams


@app.command("submit")
@exit_on_error
def submit(
    ctx: typer.Context,
    preset: str = typer.Option(..., "--preset", help="Factory preset (e.g. deberta-base)."),
    since: str | None = typer.Option(
        None,
        "--since",
        help=(
            "ISO-8601 lower bound for the vault corpus window "
            "(e.g. 2026-01-01T00:00:00Z). Omit to include from the start."
        ),
    ),
    until: str | None = typer.Option(
        None,
        "--until",
        help="ISO-8601 upper bound for the vault corpus window. Omit for 'now'.",
    ),
    epochs: int | None = typer.Option(None, "--epochs", min=1),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1),
    learning_rate: float | None = typer.Option(None, "--learning-rate"),
) -> None:
    """Submit a new training run. Returns the run id.

    GPUB-458: the dataset is built server-side from this tenant's own vault
    (source='vault', defaulted by the gateway so we omit it). `--since`/`--until`
    narrow that corpus window; there is no client-supplied dataset URL. Per-run
    tunables nest under `hyperparams` (the gateway shallow-merges over the preset
    defaults). TrainingRunCreate is extra='forbid', so no other top-level keys.
    """
    out = _output(ctx)
    body: dict = {"preset": preset}
    if since:
        body["since"] = since
    if until:
        body["until"] = until
    hyperparams = _build_hyperparams(epochs, batch_size, learning_rate)
    if hyperparams:
        body["hyperparams"] = hyperparams

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
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """Download the model artefact produced by a successful run.

    The gateway never proxies bytes: GET /training/runs/{id}/download returns a
    JSON envelope {url, expires_in, sha256, size_bytes} with a pre-signed
    storage URL. We fetch that URL directly (it carries its own signature — no
    Authorization header, which would otherwise break the signature on
    S3/R2) and stream it to disk, verifying the sha256 the gateway pinned.
    """
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

    # 1) Authenticated, workspace-scoped call to mint the signed URL.
    with _client(ctx) as client:
        envelope = client.request(
            "GET", f"/training/runs/{run_id}/download",
            extra_headers=cfg.workspace_headers(workspace),
        )
    signed_url = envelope.get("url") if isinstance(envelope, dict) else None
    if not signed_url:
        from gpubox_cli.client import GPUBoxError

        raise GPUBoxError("download response did not include a signed url")
    expected_sha = envelope.get("sha256") if isinstance(envelope, dict) else None

    # 2) Stream the pre-signed URL straight to disk. No Authorization header —
    #    the signature is in the URL; an extra bearer token breaks S3/R2 sigv4.
    headers = {"User-Agent": USER_AGENT, "Accept": "application/octet-stream"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    try:
        with httpx.stream(
            "GET", signed_url, headers=headers, timeout=600.0, follow_redirects=True
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                from gpubox_cli.client import GPUBoxError

                raise GPUBoxError(
                    f"signed URL fetch failed: HTTP {resp.status_code}"
                )
            with output_path.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 16):
                    hasher.update(chunk)
                    fh.write(chunk)
    except httpx.HTTPError as exc:
        from gpubox_cli.client import GPUBoxError

        raise GPUBoxError(f"network error during download: {exc}") from exc

    if expected_sha and hasher.hexdigest() != expected_sha:
        from gpubox_cli.client import GPUBoxError

        output_path.unlink(missing_ok=True)
        raise GPUBoxError(
            f"sha256 mismatch: expected {expected_sha}, got {hasher.hexdigest()}"
        )

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


@app.command("presets")
@exit_on_error
def presets(ctx: typer.Context) -> None:
    """List the available training presets (name, base model, VRAM, est. GPU-seconds)."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/training/presets")
    if out.json_mode:
        emit_json(out, resp)
        return
    items = resp.get("data", []) if isinstance(resp, dict) else resp
    if not items:
        emit_text(out, "(no presets)")
        return
    for p in items:
        emit_text(
            out,
            f"{p.get('name')}  base={p.get('model_base')}  "
            f"min_vram={p.get('min_vram_gb')}GB  est={p.get('estimated_gpu_seconds')}s",
        )
