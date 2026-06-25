"""`gpb finetune ...` — GPUBox V1.5 W3 workspace-scoped user fine-tunes.

The "we train it for you" flow, scoped to your active workspace:

  gpb finetune create  --preset qwen32b-lora-r16 --dataset gpubox://...   # submit a run
  gpb finetune status  <run_id>                                          # run lifecycle state
  gpb finetune list                                                      # runs (or --adapters)
  gpb finetune use     <hosted_model_name>                               # pin as workspace default

`create`/`status`/`list` drive the existing training + LoRA-registry surface;
`use` pins a hosted fine-tune as the active workspace's default chat model
(PUT /finetune/active). Every call sends the pinned active workspace as the
X-GPUBox-Workspace header (mirrors `gpb workspace use`), so the whole flow is
workspace-scoped: a fine-tune created/selected in workspace A is invisible in B.

Path style: the resolved base_url already ends in /v1, so command paths are
written WITHOUT the /v1 prefix (matching the rest of the CLI).
"""
from __future__ import annotations

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Workspace-scoped user fine-tunes (V1.5 W3).")

# Same config key + header the `gpb workspace use` command writes/reads.
_ACTIVE_WORKSPACE_KEY = "active_workspace"
WORKSPACE_HEADER = "X-GPUBox-Workspace"

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


def _active_workspace() -> str | None:
    settings = cfg.load_settings()
    val = settings.extra.get(_ACTIVE_WORKSPACE_KEY)
    return str(val) if val else None


def _workspace_headers(override: str | None) -> dict | None:
    """The X-GPUBox-Workspace header from --workspace override, else the pinned
    active workspace. None when neither is set (server defaults to Default)."""
    ws = override or _active_workspace()
    return {WORKSPACE_HEADER: ws} if ws else None


# ---------------------------------------------------------------------------
# create — submit a training run (the "we train it for you" entry point)
# ---------------------------------------------------------------------------


@app.command("create", help="Submit a fine-tune training run in the active workspace.")
@exit_on_error
def create(
    ctx: typer.Context,
    preset: str = typer.Option(..., "--preset", help="Training preset (e.g. qwen32b-lora-r16)."),
    dataset: str = typer.Option(
        ..., "--dataset", help="Dataset URI (s3://, https://, gpubox://...)."
    ),
    name: str | None = typer.Option(None, "--name", help="Optional run label."),
    epochs: int | None = typer.Option(None, "--epochs", min=1),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1),
    learning_rate: float | None = typer.Option(None, "--learning-rate"),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """Submit a new fine-tune training run. Returns the run id."""
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
        resp = client.request(
            "POST", "/training/runs", json_body=body, idempotent=True,
            extra_headers=_workspace_headers(workspace),
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    rid = resp.get("id") if isinstance(resp, dict) else None
    emit_text(out, f"submitted fine-tune run: {rid}" if rid else str(resp))


# ---------------------------------------------------------------------------
# list — runs (default) or registered adapters (--adapters)
# ---------------------------------------------------------------------------


@app.command("list", help="List fine-tune runs (or --adapters) in the active workspace.")
@exit_on_error
def list_finetunes(
    ctx: typer.Context,
    adapters: bool = typer.Option(
        False, "--adapters", help="List registered LoRA adapters instead of runs."
    ),
    status: str | None = typer.Option(None, "--status", help="Filter runs by status."),
    limit: int = typer.Option(20, "--limit", "-n"),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """List fine-tune training runs, or registered adapters with --adapters."""
    out = _output(ctx)
    headers = _workspace_headers(workspace)
    with _client(ctx) as client:
        if adapters:
            resp = client.request("GET", "/lora/adapters", extra_headers=headers)
            if out.json_mode:
                emit_json(out, resp)
                return
            items = resp.get("data", []) if isinstance(resp, dict) else []
            if not items:
                emit_text(out, "(no adapters)")
                return
            for item in items:
                emit_text(
                    out,
                    f"{item.get('name','?'):<24} v{item.get('version','?'):<4} "
                    f"{item.get('status','?'):<12} {item.get('family','?')}",
                )
            return
        params: dict = {"limit": limit}
        if status:
            params["status"] = status
        resp = client.request(
            "GET", "/training/runs", params=params, extra_headers=headers
        )
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


# ---------------------------------------------------------------------------
# status — one-shot run lifecycle state
# ---------------------------------------------------------------------------


@app.command("status", help="Show the lifecycle state of a fine-tune run.")
@exit_on_error
def status_finetune(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="Training run id."),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """One-shot status of a fine-tune run."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request(
            "GET", f"/training/runs/{run_id}",
            extra_headers=_workspace_headers(workspace),
        )
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"id:       {resp.get('id')}")
    emit_text(out, f"status:   {resp.get('status')}")
    if "progress" in resp:
        emit_text(out, f"progress: {resp.get('progress')}")
    if "metrics" in resp:
        emit_text(out, f"metrics:  {resp.get('metrics')}")


# ---------------------------------------------------------------------------
# use — pin a hosted fine-tune as the active workspace's default chat model
# ---------------------------------------------------------------------------


@app.command("use", help="Pin a hosted fine-tune as the active workspace's default chat model.")
@exit_on_error
def use_finetune(
    ctx: typer.Context,
    hosted_model_name: str = typer.Argument(
        ..., help="The hosted_model_name to pin (the `lora:<name>` model name)."
    ),
    clear: bool = typer.Option(
        False, "--clear", help="Clear the pin instead (ignore the name argument)."
    ),
    show: bool = typer.Option(
        False, "--show", help="Show the current pin instead of changing it."
    ),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Override the active workspace for this command."
    ),
) -> None:
    """Pin / clear / show the active workspace's active fine-tune.

    On success the server echoes the `chat_model` string (`lora:<name>`) you'd
    otherwise send by hand; once pinned, plain chat in this workspace defaults
    to the fine-tune automatically.
    """
    out = _output(ctx)
    headers = _workspace_headers(workspace)
    with _client(ctx) as client:
        if show:
            resp = client.request("GET", "/finetune/active", extra_headers=headers)
        elif clear:
            resp = client.request("DELETE", "/finetune/active", extra_headers=headers)
        else:
            resp = client.request(
                "PUT", "/finetune/active",
                json_body={"hosted_model_name": hosted_model_name},
                extra_headers=headers,
            )
    if out.json_mode:
        emit_json(out, resp)
        return
    name = resp.get("hosted_model_name") if isinstance(resp, dict) else None
    chat_model = resp.get("chat_model") if isinstance(resp, dict) else None
    if name:
        emit_text(out, f"active fine-tune: {name}  (chat with model={chat_model})")
    else:
        emit_text(out, "no active fine-tune pinned (chatting on the base model)")
