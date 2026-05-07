"""`gpb billing ...` — wallet balance, top-ups, history.

Round-table lock #5: top-ups open the hosted Stripe / Paystack checkout
URL in the browser instead of capturing card details — PCI-safer and the
right primitive. NGN top-ups are first-class for the NG market.

Backend contracts (verified against gpubox-gateway prod):

* GET  /v1/billing/balance              → balance_pence + recent_topups
* POST /v1/billing/checkout-sessions    → Stripe (amount_pence, client_idempotency_key)
* POST /v1/billing/paystack/initialize  → Paystack (amount_kobo, client_idempotency_key)

The gateway requires ``client_idempotency_key`` in the BODY (8-128 chars,
namespaced server-side) — different from a generic Idempotency-Key
header. We generate one per CLI invocation; reusing the same key for the
same amount safely returns the same checkout URL.
"""

from __future__ import annotations

import contextlib
import uuid
import webbrowser

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_error, emit_json, emit_text

app = typer.Typer(no_args_is_help=True, help="Wallet, top-ups, usage.")

#: Hosted dashboard fallback when Stripe/Paystack don't return a URL.
DASHBOARD_BILLING_URL = "https://gpubox.ai/dashboard/billing"


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


def _new_idempotency_key() -> str:
    """Generate an 8+ char body-level idempotency key per gateway spec."""
    return f"gpb-cli-{uuid.uuid4()}"


def _format_minor_units(amount_minor: int, currency: str) -> str:
    """Convert provider-native minor units to a human-readable major-unit
    string with the right symbol.

    Per Codex review: ``charged_amount_minor`` is pence for Stripe (GBP)
    and kobo for Paystack (NGN). Printing the raw number is misleading
    (e.g. ``1000000 NGN`` looks like ten million naira when it's ten
    thousand). Both currencies use a 100-unit subdivision so /100 works.
    """
    ccy = (currency or "").lower()
    major = amount_minor / 100
    symbols = {"gbp": "£", "ngn": "₦"}
    sym = symbols.get(ccy, "")
    return f"{sym}{major:,.2f} {ccy.upper()}".strip() if not sym else f"{sym}{major:,.2f}"


def _format_topup_line(t: dict) -> str:
    """Single source of truth for the human-readable top-up row.

    Both ``balance`` and ``history`` reuse this to keep formatting
    consistent. Charged amount is rendered in its native currency
    (kobo→naira, pence→pounds); credited amount is GBP-pence per the
    gateway ledger contract.
    """
    ts = t.get("created_at", "?")
    provider = t.get("provider", "?")
    charged_minor = t.get("charged_amount_minor", 0)
    charged_ccy = t.get("charged_currency", "")
    credited = t.get("credited_amount_pence", 0)
    status = t.get("status", "?")
    return (
        f"{ts}  {provider:<8}  "
        f"charged {_format_minor_units(charged_minor, charged_ccy)}  "
        f"→ credited £{credited / 100:.2f}  {status}"
    )


@app.command("balance")
@exit_on_error
def balance(ctx: typer.Context) -> None:
    """Show current wallet balance + recent top-ups."""
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/billing/balance")

    if out.json_mode:
        emit_json(out, resp)
        return

    if not isinstance(resp, dict):
        emit_text(out, str(resp))
        return

    # Server returns balance in pence (GBP) — render as £x.yz for humans.
    pence = resp.get("balance_pence")
    available = resp.get("available_pence")
    plan = resp.get("plan", "?")
    if isinstance(pence, int):
        emit_text(out, f"balance:   £{pence / 100:.2f}")
    if isinstance(available, int) and available != pence:
        emit_text(out, f"available: £{available / 100:.2f}  (after reservations)")
    emit_text(out, f"plan:      {plan}")
    topups = resp.get("recent_topups") or []
    if topups:
        emit_text(out, "")
        emit_text(out, "recent top-ups:")
        for t in topups[:5]:
            emit_text(out, f"  {_format_topup_line(t)}")


@app.command("topup")
@exit_on_error
def topup(
    ctx: typer.Context,
    amount_gbp: float | None = typer.Option(
        None, "--amount-gbp", help="Top up in GBP via Stripe checkout."
    ),
    amount_ngn: float | None = typer.Option(
        None, "--amount-ngn", help="Top up in NGN via Paystack."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Print the checkout URL instead of opening it."
    ),
) -> None:
    """Open a hosted Stripe (GBP) or Paystack (NGN) checkout for a top-up."""
    out = _output(ctx)
    if (amount_gbp is None) == (amount_ngn is None):
        emit_error(out, "pass exactly one of --amount-gbp or --amount-ngn")
        raise typer.Exit(2)

    idem = _new_idempotency_key()

    if amount_gbp is not None:
        # Stripe path — convert pounds → integer pence; gateway validates range.
        body = {
            "amount_pence": int(round(amount_gbp * 100)),
            "currency": "gbp",
            "client_idempotency_key": idem,
        }
        path = "/billing/checkout-sessions"
    else:
        # Paystack path — convert naira → integer kobo (1 NGN = 100 kobo).
        body = {
            "amount_kobo": int(round((amount_ngn or 0) * 100)),
            "currency": "ngn",
            "client_idempotency_key": idem,
        }
        path = "/billing/paystack/initialize"

    with _client(ctx) as client:
        # idempotent=False here: we already pass client_idempotency_key in body
        # per gateway contract. Adding the header would be redundant and the
        # server forbids unknown fields (extra="forbid" on the model).
        resp = client.request("POST", path, json_body=body)

    # Stripe returns "url"; Paystack returns "authorization_url".
    checkout_url: str | None = None
    if isinstance(resp, dict):
        checkout_url = resp.get("url") or resp.get("authorization_url")

    if not checkout_url:
        emit_error(out, "server did not return a checkout URL")
        if out.json_mode:
            emit_json(out, resp)
        raise typer.Exit(1)

    if out.json_mode:
        emit_json(out, resp)
        return

    emit_text(out, f"checkout: {checkout_url}")
    if not no_browser:
        with contextlib.suppress(Exception):  # pragma: no cover
            webbrowser.open_new_tab(checkout_url)


@app.command("history")
@exit_on_error
def history(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=100),
) -> None:
    """Show recent top-ups (alias for `balance` — same source).

    The server doesn't expose a dedicated /history endpoint today; we
    re-read /balance and surface ``recent_topups``. When a richer history
    endpoint ships, we'll route here without changing the user contract.
    """
    out = _output(ctx)
    with _client(ctx) as client:
        resp = client.request("GET", "/billing/balance")

    if out.json_mode:
        emit_json(out, resp)
        return

    items = resp.get("recent_topups", []) if isinstance(resp, dict) else []
    if not items:
        emit_text(out, "(no history)")
        return
    for item in items[:limit]:
        emit_text(out, _format_topup_line(item))
