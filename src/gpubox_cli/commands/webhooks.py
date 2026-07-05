"""`gpb webhooks ...` — manage webhook subscriptions + observe deliveries.

Webhook management is SERVICE-KEY-ONLY on the gateway (require_service_key):
these commands need a `gpb_live_*` service API key, not a browser session. The
delivered events are HMAC-signed (X-GPUBox-Signature: t=<unix>,v1=<hex>); the
`signing_secret` is returned ONCE at create/rotate — store it.

Verbs mirror the gateway's /v1/webhooks surface (GPUB-455 / GPUB-610):
  create · list · get · delete · rotate-secret · deliveries · test · replay
Only the training.run.* event types are subscribable today (the only ones with
a producer); `test` fires a synthetic webhook.test delivery with no GPU spend.
"""

from __future__ import annotations

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Webhook subscriptions + delivery monitoring.")

# The subscribable event types (the gateway rejects anything else with 400
# unknown_event_type). Kept here for --help discoverability; the gateway is the
# source of truth.
EVENT_TYPES = [
    "training.run.created",
    "training.run.running",
    "training.run.succeeded",
    "training.run.failed",
    "training.run.cancelled",
]


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


@app.command("create")
@exit_on_error
def create(
    ctx: typer.Context,
    url: str = typer.Option(..., "--url", help="Public HTTPS receiver (private/localhost rejected by SSRF)."),
    events: list[str] = typer.Option(
        None, "--event", "-e",
        help="Event type to subscribe to (repeatable). Default: all training.run.* types.",
    ),
) -> None:
    """Register a webhook subscription. Prints the signing_secret ONCE — store it."""
    out = _output(ctx)
    event_types = list(events) if events else list(EVENT_TYPES)
    body = {"url": url, "event_types": event_types}
    with _client(ctx) as client:
        resp = client.request("POST", "/webhooks", json_body=body)
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"webhook: {resp.get('id')}")
    emit_text(out, f"events:  {', '.join(resp.get('event_types', []))}")
    emit_text(out, f"SIGNING SECRET (shown once): {resp.get('signing_secret')}")


@app.command("list")
@exit_on_error
def list_webhooks(ctx: typer.Context) -> None:
    """List this tenant's webhook subscriptions."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/webhooks")
    if out.json_mode:
        emit_json(out, resp)
        return
    items = resp if isinstance(resp, list) else resp.get("data", [])
    if not items:
        emit_text(out, "(no webhooks)")
        return
    for w in items:
        active = "active" if w.get("active") else "inactive"
        emit_text(out, f"{w.get('id')}  {active}  {w.get('url')}  [{', '.join(w.get('event_types', []))}]")


@app.command("get")
@exit_on_error
def get(ctx: typer.Context, webhook_id: str = typer.Argument(...)) -> None:
    """Fetch one webhook subscription."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", f"/webhooks/{webhook_id}")
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, str(resp))


@app.command("delete")
@exit_on_error
def delete(ctx: typer.Context, webhook_id: str = typer.Argument(...)) -> None:
    """Deactivate (soft-delete) a webhook subscription."""
    out = _output(ctx)
    with _client(ctx) as client:
        client.request("DELETE", f"/webhooks/{webhook_id}")
    emit_text(out, f"deleted: {webhook_id}")


@app.command("rotate-secret")
@exit_on_error
def rotate_secret(ctx: typer.Context, webhook_id: str = typer.Argument(...)) -> None:
    """Rotate the signing secret. Prints the NEW secret ONCE (old works 24h)."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("POST", f"/webhooks/{webhook_id}/rotate-secret")
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"NEW SIGNING SECRET (shown once): {resp.get('signing_secret')}")


@app.command("deliveries")
@exit_on_error
def deliveries(
    ctx: typer.Context,
    webhook_id: str = typer.Argument(...),
    limit: int = typer.Option(20, "--limit", "-n", min=1, max=100),
    cursor: str | None = typer.Option(None, "--cursor", help="Opaque next_cursor from a prior page."),
) -> None:
    """Delivery history for a webhook: never-fired vs backing-off vs dead-lettered."""
    out = _output(ctx)
    params: dict = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    with _client(ctx) as client:
        resp = client.request("GET", f"/webhooks/{webhook_id}/deliveries", params=params)
    if out.json_mode:
        emit_json(out, resp)
        return
    for d in resp.get("data", []):
        err = f"  err={d['last_attempt_error']}" if d.get("last_attempt_error") else ""
        emit_text(
            out,
            f"{d.get('event_type')}  {d.get('status')}  attempts={d.get('attempt_count')}  "
            f"{d.get('event_id')}{err}",
        )
    if resp.get("next_cursor"):
        emit_text(out, f"(next page: --cursor {resp['next_cursor']})")
    if not resp.get("data"):
        emit_text(out, "(no deliveries)")


@app.command("test")
@exit_on_error
def test(ctx: typer.Context, webhook_id: str = typer.Argument(...)) -> None:
    """Fire ONE synthetic webhook.test delivery (validate receiver + signature, no GPU spend)."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("POST", f"/webhooks/{webhook_id}/test")
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"test fired: event_id={resp.get('event_id')}")


@app.command("replay")
@exit_on_error
def replay(
    ctx: typer.Context,
    webhook_id: str = typer.Argument(...),
    event_id: str = typer.Argument(..., help="The event_id of a dead-lettered/pending delivery."),
) -> None:
    """Re-enqueue a dead-lettered delivery for one more retry cycle."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("POST", f"/webhooks/{webhook_id}/deliveries/{event_id}/replay")
    if out.json_mode:
        emit_json(out, resp)
        return
    emit_text(out, f"replay queued: {event_id}")
