"""Billing command happy-paths.

Tests pinned against the REAL gateway contract (verified vs
gpubox-gateway/app/billing/). Per Codex review: don't lock in mock
shapes that diverge from the server.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app

BASE = cfg.DEFAULT_API_URL


@pytest.fixture(autouse=True)
def authed(fake_api_key: str) -> None:
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key, base_url=BASE))


@respx.mock
def test_balance_renders_pence_as_pounds(runner: CliRunner) -> None:
    respx.get(f"{BASE}/billing/balance").mock(
        return_value=httpx.Response(
            200,
            json={
                "balance_pence": 1234,
                "available_pence": 1000,
                "currency": "gbp",
                "plan": "payg",
                "recent_topups": [],
            },
        )
    )
    result = runner.invoke(app, ["billing", "balance"])
    assert result.exit_code == 0
    assert "£12.34" in result.stdout
    assert "£10.00" in result.stdout
    assert "payg" in result.stdout


@respx.mock
def test_topup_gbp_uses_stripe_checkout_sessions(runner: CliRunner) -> None:
    """Body must contain client_idempotency_key + amount_pence (not generic header)."""
    route = respx.post(f"{BASE}/billing/checkout-sessions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "cs_test_1",
                "url": "https://checkout.stripe.com/pay/cs_test_1",
                "amount_pence": 1000,
                "currency": "gbp",
            },
        )
    )
    result = runner.invoke(
        app, ["billing", "topup", "--amount-gbp", "10", "--no-browser"]
    )
    assert result.exit_code == 0, result.stderr
    assert "checkout.stripe.com" in result.stdout

    sent = route.calls.last.request
    body = json.loads(sent.read())
    assert body["amount_pence"] == 1000
    assert body["currency"] == "gbp"
    assert "client_idempotency_key" in body
    assert len(body["client_idempotency_key"]) >= 8


@respx.mock
def test_topup_ngn_uses_paystack_initialize(runner: CliRunner) -> None:
    """NGN path uses Paystack endpoint + amount_kobo + body idempotency key."""
    route = respx.post(f"{BASE}/billing/paystack/initialize").mock(
        return_value=httpx.Response(
            200,
            json={
                "authorization_url": "https://checkout.paystack.com/abc",
                "reference": "ref_1",
            },
        )
    )
    result = runner.invoke(
        app, ["billing", "topup", "--amount-ngn", "10000", "--no-browser"]
    )
    assert result.exit_code == 0
    assert "paystack.com" in result.stdout
    body = json.loads(route.calls.last.request.read())
    assert body["amount_kobo"] == 1000000  # 10,000 NGN = 1,000,000 kobo
    assert body["currency"] == "ngn"
    assert "client_idempotency_key" in body


def test_topup_requires_one_currency(runner: CliRunner) -> None:
    result = runner.invoke(app, ["billing", "topup"])
    assert result.exit_code == 2
