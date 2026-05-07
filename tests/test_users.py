"""Users + OIDC contract tests — verified vs gateway shapes."""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gpubox_cli import config as cfg
from gpubox_cli.main import app

BASE = cfg.DEFAULT_API_URL
TENANT = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def authed(fake_api_key: str) -> None:
    cfg.upsert_profile("default", cfg.Profile(api_key=fake_api_key, base_url=BASE))


def test_invite_without_tenant_id_fails_with_clear_error(runner: CliRunner) -> None:
    """Per Codex review: tenant-scoped paths need tenant_id. Don't 404 silently."""
    result = runner.invoke(app, ["users", "invite", "alice@acme.com"])
    assert result.exit_code == 2
    assert "tenant_id" in result.stderr or "tenant" in result.stderr.lower()


@respx.mock
def test_invite_uses_tenant_scoped_path(runner: CliRunner) -> None:
    route = respx.post(f"{BASE}/tenants/{TENANT}/users").mock(
        return_value=httpx.Response(201, json={"id": "u_1", "email": "a@b.co"})
    )
    result = runner.invoke(
        app, ["users", "invite", "a@b.co", "--role", "viewer", "--tenant", TENANT]
    )
    assert result.exit_code == 0, result.stderr
    body = json.loads(route.calls.last.request.read())
    assert body["email"] == "a@b.co"
    assert body["role"] == "viewer"


@respx.mock
def test_oidc_create_sends_redirect_uris_list(runner: CliRunner) -> None:
    """Gateway requires redirect_uris as a LIST — Codex flagged this."""
    route = respx.post(f"{BASE}/oidc/clients").mock(
        return_value=httpx.Response(
            201,
            json={"client_id": "cli_1", "client_secret": "secret-shown-once"},
        )
    )
    result = runner.invoke(
        app,
        [
            "users",
            "oidc",
            "create",
            "--name",
            "Acme",
            "--redirect-uri",
            "https://acme.example/cb",
            "--redirect-uri",
            "https://acme.example/cb2",
        ],
    )
    assert result.exit_code == 0
    body = json.loads(route.calls.last.request.read())
    assert isinstance(body["redirect_uris"], list)
    assert body["redirect_uris"] == [
        "https://acme.example/cb",
        "https://acme.example/cb2",
    ]
    # Default client_type should be sent.
    assert body["client_type"] == "confidential"
    # Client secret in stdout-text should NOT leak — only via --json.
    assert "secret-shown-once" not in result.stdout


@respx.mock
def test_users_list_renders_user_id_field(runner: CliRunner) -> None:
    """Per Codex round-2: gateway returns ``user_id`` (not ``id``).

    Pin the plain-text render so a future server-side rename surfaces here
    before the CLI ships a wall of '?' for user IDs.
    """
    respx.get(f"{BASE}/tenants/{TENANT}/users").mock(
        return_value=httpx.Response(
            200,
            json={
                "users": [
                    {
                        "user_id": "u_abc123",
                        "email": "alice@acme.com",
                        "role": "admin",
                        "status": "active",
                    }
                ]
            },
        )
    )
    result = runner.invoke(app, ["users", "list", "--tenant", TENANT])
    assert result.exit_code == 0
    assert "u_abc123" in result.stdout
    assert "?" not in result.stdout.split("u_abc123")[0]  # no ? before the id


@respx.mock
def test_oidc_list_reads_clients_key(runner: CliRunner) -> None:
    """Server returns ``clients`` not ``items``."""
    respx.get(f"{BASE}/oidc/clients").mock(
        return_value=httpx.Response(
            200,
            json={"clients": [{"client_id": "cli_1", "name": "Acme"}]},
        )
    )
    result = runner.invoke(app, ["users", "oidc", "list"])
    assert result.exit_code == 0
    assert "cli_1" in result.stdout
    assert "Acme" in result.stdout
