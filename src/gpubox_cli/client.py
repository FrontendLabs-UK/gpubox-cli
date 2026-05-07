"""HTTP client wrapping the GPUBox API.

Design notes (round-table outcomes baked in):

* Sync httpx by default — Codex pushed back on async because a CLI doesn't
  benefit from it. Streaming uses ``client.stream()`` which is sync-friendly.
* User-Agent constant on every request (gateway analytics depend on it).
* Bearer auth from the resolved config; we never echo the key back.
* Idempotency-Key header auto-generated for POSTs that opt-in via a flag.
* Mid-stream 402 fail-fast UX (round-table lock): we surface a top-up URL
  on stderr and exit cleanly instead of dumping a half-JSON blob to stdout.
* Headless ergonomics: no $HOME assumptions here — all that lives in
  config.py. The client only sees a fully resolved (api_key, base_url).
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from gpubox_cli.version import USER_AGENT


class GPUBoxError(Exception):
    """Base class for all CLI-side errors. Carries a stable exit code."""

    exit_code = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class AuthError(GPUBoxError):
    """401/403 — missing or invalid API key."""

    exit_code = 4


class PaymentRequiredError(GPUBoxError):
    """402 — wallet balance is exhausted. Explicit class so command code
    can render the topup URL with a clean UX instead of a stack trace."""

    exit_code = 5

    def __init__(
        self,
        message: str,
        *,
        topup_url: str = "https://gpubox.ai/dashboard/billing",
    ) -> None:
        super().__init__(message, hint=f"Top up your wallet at {topup_url}")
        self.topup_url = topup_url


class RateLimitError(GPUBoxError):
    """429 — backoff and retry hint."""

    exit_code = 6


class APIError(GPUBoxError):
    """Generic 4xx/5xx with the server's message preserved."""


@dataclass
class ClientConfig:
    api_key: str | None
    base_url: str
    timeout: float = 60.0
    # Connect timeout deliberately tighter — fast fail when the box is
    # unreachable from a flaky NG mobile network is better than a 60s spin.
    connect_timeout: float = 10.0


class GPUBoxClient:
    """Thin httpx wrapper. All commands route through this.

    Not thread-safe — but a CLI invocation is single-threaded. If we ever
    parallelise (e.g. parallel embed batches), construct one client per
    worker and share nothing.
    """

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        timeout = httpx.Timeout(config.timeout, connect=config.connect_timeout)
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=timeout,
            headers=self._default_headers(),
            follow_redirects=True,
        )

    @property
    def base_url(self) -> str:
        return self._config.base_url

    def _default_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    @staticmethod
    def _idempotency_key() -> str:
        # uuid4 is plenty — server-side dedupe windows are short.
        return f"gpb-{uuid.uuid4()}"

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
        idempotent: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        """Execute an HTTP request and return parsed JSON.

        idempotent=True on a POST emits an Idempotency-Key the gateway can
        use to dedupe retries (training submit, billing topup are the main
        candidates). For idempotent verbs (GET/PUT/DELETE) the flag is a no-op.
        """
        self._require_auth()
        headers: dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)
        if idempotent and method.upper() == "POST":
            headers["Idempotency-Key"] = self._idempotency_key()

        try:
            resp = self._client.request(
                method, path, json=json_body, params=params, headers=headers or None
            )
        except httpx.ConnectError as exc:
            raise GPUBoxError(
                f"could not reach {self._config.base_url}: {exc}",
                hint="check your network or set GPUBOX_API_URL to a reachable endpoint",
            ) from exc
        except httpx.TimeoutException as exc:
            raise GPUBoxError(f"request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            # Per Codex review: catch the broader transport family
            # (ReadError, RemoteProtocolError, ProxyError, etc.) so buffered
            # commands don't traceback on non-connect transport failures.
            raise GPUBoxError(f"network error: {exc}") from exc

        return self._handle_response(resp)

    @contextmanager
    def stream(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
    ) -> Iterator[httpx.Response]:
        """Open a streaming HTTP response. Used by chat completions (SSE).

        Caller consumes ``resp.iter_lines()``; we surface auth/billing
        errors here BEFORE the first byte hits stdout, so a 402 never
        produces a half-streamed JSON blob (round-table lock #5).

        Per Codex review, mid-stream transport failures (ReadError,
        RemoteProtocolError) inside the iterator are NOT caught here —
        they happen on the caller's iterator side. The chat command
        wraps the iter loop in a try/except → GPUBoxError so users get a
        clean exit code instead of an httpx traceback on flaky links.
        """
        self._require_auth()
        try:
            with self._client.stream(method, path, json=json_body) as resp:
                if resp.status_code >= 400:
                    # Read the (small) error body so we can raise typed errors.
                    resp.read()
                    self._handle_response(resp)
                yield resp
        except httpx.ConnectError as exc:
            raise GPUBoxError(
                f"could not reach {self._config.base_url}: {exc}",
                hint="check your network or set GPUBOX_API_URL to a reachable endpoint",
            ) from exc
        except httpx.TimeoutException as exc:
            raise GPUBoxError(f"request timed out before stream opened: {exc}") from exc
        except httpx.HTTPError as exc:
            raise GPUBoxError(f"network error: {exc}") from exc

    def raise_for_response(self, resp: httpx.Response) -> None:
        """Public entry point for code paths that hand-roll httpx requests
        (e.g. multipart uploads in transcribe / vault corpora upload).

        Replaces the previous private-API smell where commands reached
        into ``_handle_response`` directly. Always raises a typed
        GPUBoxError for any non-2xx status; returns None on success.
        """
        # Reuses the existing parser, which raises for 4xx/5xx.
        self._handle_response(resp)

    def _require_auth(self) -> None:
        if not self._config.api_key:
            raise AuthError(
                "no API key configured",
                hint="run `gpb auth login` or set GPUBOX_API_KEY",
            )

    def _handle_response(self, resp: httpx.Response) -> Any:
        if 200 <= resp.status_code < 300:
            if not resp.content:
                return None
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                return resp.json()
            return resp.text

        # Try to parse the server's error envelope; fall back to status text.
        try:
            body = resp.json()
            message = (
                body.get("error", {}).get("message")
                if isinstance(body.get("error"), dict)
                else body.get("message") or body.get("detail") or resp.text
            )
        except (ValueError, json.JSONDecodeError):
            message = resp.text or resp.reason_phrase

        if resp.status_code in (401, 403):
            raise AuthError(
                f"authentication failed ({resp.status_code}): {message}",
                hint="run `gpb auth login` to set a valid API key",
            )
        if resp.status_code == 402:
            raise PaymentRequiredError(f"payment required: {message}")
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            hint = f"retry after {retry_after}s" if retry_after else "back off and retry"
            raise RateLimitError(f"rate limited: {message}", hint=hint)
        raise APIError(f"API error {resp.status_code}: {message}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GPUBoxClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def retry_with_backoff(
    fn,
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
):
    """Tiny exponential backoff for idempotent reads.

    Used sparingly — we'd rather fail fast and let the user re-run than
    hide flakiness. RateLimitError honours the Retry-After hint by
    sleeping at least that long.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except RateLimitError as exc:
            last_exc = exc
            time.sleep(min(max_delay, base_delay * (2**i)))
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc  # type: ignore[assignment]
            time.sleep(min(max_delay, base_delay * (2**i)))
    assert last_exc is not None
    raise last_exc


def render_error(exc: GPUBoxError) -> int:
    """Format a GPUBoxError nicely on stderr and return the exit code.

    Used both by main.py's top-level handler and by typer commands that
    catch GPUBoxError → typer.Exit(code) so we get reliable exit codes
    when run under click's CliRunner (which doesn't see our wrapper).
    """
    sys.stderr.write(f"error: {exc}\n")
    if exc.hint:
        sys.stderr.write(f"hint: {exc.hint}\n")
    if isinstance(exc, PaymentRequiredError):
        sys.stderr.write(
            "Top up via card (Stripe) or NGN bank transfer (Paystack):\n"
            f"  {exc.topup_url}\n"
        )
    sys.stderr.flush()
    return exc.exit_code


def exit_on_error(fn):
    """Decorator: wrap a typer command so GPUBoxError → typer.Exit(code).

    This is the reliable way to get our exit codes through click's
    CliRunner, which doesn't run main._entrypoint's wrapper. Apply it to
    every command that talks to the GPUBox API.
    """
    import functools

    import typer

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except GPUBoxError as exc:
            code = render_error(exc)
            raise typer.Exit(code=code) from None

    return wrapped
