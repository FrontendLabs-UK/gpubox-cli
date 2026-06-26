"""Dependency-free request-shape validator against a pinned OpenAPI snapshot.

`assert_request_conforms(spec, method, path, body, query=None)` raises
`ConformanceError` when the request the CLI is about to send does not match the
gateway's declared contract. It catches exactly the four failure modes that
caused PR #12's eight drifts:

1. **Unregistered method/path** — no operation for `(method, path)` in the spec.
   This is what a PATCH-on-a-POST-only route (405) and a `/runs` sub-resource
   that does not exist (404) look like statically.
2. **Undeclared key under `additionalProperties: false`** — a key in `body` that
   the request schema does not declare, where the schema forbids extras. This is
   the Pydantic ``extra='forbid'`` 422 (`dataset`, `name`, `epochs`, ...).
3. **Missing required property** — a `required` property absent from `body`
   (missing `slug`, missing `training_run_id`).
4. (defensive) a malformed spec — a `$ref` that does not resolve.

Design choices, and why:

* **No `jsonschema` dependency.** The CLI keeps its dependency surface tiny
  (typer/httpx/rich). A full JSON-Schema engine would validate values too, but
  the drifts we are guarding are all SHAPE drifts (keys + presence), which a
  small manual resolver covers exactly. We deliberately do NOT validate value
  types/enums here — see the PR body's "validates SHAPE not semantics" caveat.
* **Path-template matching.** The CLI builds concrete paths like
  ``/assistants/abc123``; the spec stores the templated form
  ``/v1/assistants/{assistant_id}``. We match segment-by-segment, treating a
  ``{...}`` spec segment as a wildcard. The CLI's ``base_url`` carries the
  ``/v1`` prefix, so a CLI path of ``/assistants`` is matched against the spec's
  ``/v1/assistants`` by trying each configured ``api_prefix``.
* **`additionalProperties` semantics follow OpenAPI 3.1.** Only an explicit
  ``false`` makes an unknown key a hard failure. ``None`` (unset) or ``true``
  means extras are tolerated by the server, so we do not fail on them — matching
  the gateway's actual Pydantic models (``WorkspaceCreate`` /
  ``CreateAgentRequest`` are not ``extra='forbid'``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The CLI's base_url is ``https://api.gpubox.ai/v1`` (see config.DEFAULT_API_URL),
# so a CLI path of ``/assistants`` reaches the gateway route stored in the spec
# as ``/v1/assistants``. We try the path verbatim first (covers any future
# already-prefixed callers and the root-mounted /oidc/.well-known/* routes), then
# the ``/v1``-prefixed form.
API_PREFIXES: tuple[str, ...] = ("", "/v1")

_SNAPSHOT = Path(__file__).with_name("gateway-openapi.json")


class _Missing:
    """Sentinel for "the caller sent no body at all".

    Distinct from ``None`` (a JSON ``null`` body) and ``{}`` (an empty object
    body) — both of which ARE bodies the gateway sees and can reject. Without
    this distinction ``if body:`` would silently treat ``{}``/``null``/``0`` as
    "no body", letting an empty body sail past a body-less or required-body
    endpoint (Codex finding).
    """

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "<no body sent>"


MISSING: _Missing = _Missing()


class ConformanceError(AssertionError):
    """A CLI request does not conform to the pinned gateway contract.

    Subclasses ``AssertionError`` so it reads naturally in pytest output and so a
    bare ``assert_request_conforms(...)`` failure looks like any other assertion.
    """


def load_spec(path: str | Path | None = None) -> dict[str, Any]:
    """Load the pinned OpenAPI snapshot (or an explicit path, for tests)."""
    target = Path(path) if path is not None else _SNAPSHOT
    with target.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalise_path(path: str) -> str:
    """Strip query string and a trailing slash (except the root)."""
    path = path.split("?", 1)[0]
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path


def _path_matches(spec_path: str, concrete_path: str) -> bool:
    """Match a concrete CLI path against a templated spec path, segment-wise.

    ``/v1/assistants/{assistant_id}`` matches ``/v1/assistants/abc123`` because
    the ``{assistant_id}`` segment is a wildcard. Segment counts must be equal;
    a wildcard never matches an empty segment.
    """
    spec_segs = spec_path.strip("/").split("/")
    concrete_segs = concrete_path.strip("/").split("/")
    if len(spec_segs) != len(concrete_segs):
        return False
    for spec_seg, concrete_seg in zip(spec_segs, concrete_segs, strict=True):
        if spec_seg.startswith("{") and spec_seg.endswith("}"):
            if concrete_seg == "":
                return False  # a path param must have a value
            continue
        if spec_seg != concrete_seg:
            return False
    return True


def resolve_operation(
    spec: dict[str, Any], method: str, path: str
) -> dict[str, Any] | None:
    """Return the operation object for ``(method, path)`` or ``None`` if none is
    registered. Tries each ``API_PREFIXES`` candidate so the CLI's prefix-less
    paths line up with the spec's ``/v1``-prefixed routes."""
    method_l = method.lower()
    concrete = _normalise_path(path)
    paths = spec.get("paths", {})
    for prefix in API_PREFIXES:
        candidate = prefix + concrete if prefix else concrete
        # Fast path: an exact templated match (e.g. literal paths with no params).
        op = paths.get(candidate)
        if op is not None and method_l in op:
            return op[method_l]
        # Slow path: segment-wise template match for parameterised routes.
        for spec_path, item in paths.items():
            if method_l in item and _path_matches(spec_path, candidate):
                return item[method_l]
    return None


def _deref(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve a (possibly chained) local ``$ref`` into components/schemas.

    Only local refs (``#/components/schemas/Name``) are supported — that is all
    a FastAPI-generated spec emits. A non-resolving ref raises ConformanceError
    rather than KeyError so a stale/corrupt snapshot surfaces as a clear failure.
    """
    seen: set[str] = set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if ref in seen:
            raise ConformanceError(f"circular $ref while resolving {ref!r}")
        seen.add(ref)
        if not ref.startswith("#/"):
            raise ConformanceError(f"unsupported non-local $ref {ref!r}")
        node: Any = spec
        for part in ref.lstrip("#/").split("/"):
            if not isinstance(node, dict) or part not in node:
                raise ConformanceError(f"could not resolve $ref {ref!r}")
            node = node[part]
        schema = node
    return schema


def _request_schema(
    spec: dict[str, Any], operation: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    """Pull the ``application/json`` request-body schema, dereferenced.

    Returns ``(schema, body_required)``. ``schema`` is ``None`` when the
    operation declares no JSON request body — a body-less call (e.g.
    ``POST /argus/agents/{id}/run``) is conformant as long as the CLI sends no
    body, which the caller checks. ``body_required`` reflects the OpenAPI
    ``requestBody.required`` flag so omitting a required body is a drift.
    """
    request_body = operation.get("requestBody")
    if not request_body:
        return None, False
    body_required = bool(request_body.get("required", False))
    json_media = request_body.get("content", {}).get("application/json")
    if not json_media or "schema" not in json_media:
        return None, body_required
    return _deref(spec, json_media["schema"]), body_required


def _merge_allof(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Flatten a top-level ``allOf`` into a single properties/required view.

    FastAPI rarely emits ``allOf`` for request bodies, but composing it keeps the
    validator honest if the gateway ever splits a model into a base + mixin.
    Later members win on ``additionalProperties`` only when they set ``false``
    (forbidding extras is never silently relaxed by a sibling).
    """
    if "allOf" not in schema:
        return schema
    merged_props: dict[str, Any] = dict(schema.get("properties", {}))
    merged_required: list[str] = list(schema.get("required", []))
    additional = schema.get("additionalProperties")
    for member in schema["allOf"]:
        member = _deref(spec, member)
        member = _merge_allof(spec, member)
        merged_props.update(member.get("properties", {}))
        merged_required.extend(member.get("required", []))
        if member.get("additionalProperties") is False:
            additional = False
    out = dict(schema)
    out.pop("allOf", None)
    out["properties"] = merged_props
    out["required"] = merged_required
    if additional is not None:
        out["additionalProperties"] = additional
    return out


def assert_request_conforms(
    spec: dict[str, Any],
    method: str,
    path: str,
    body: Any | None | _Missing = MISSING,
    query: dict[str, Any] | None = None,  # noqa: ARG001 (reserved; see below)
) -> None:
    """Assert the request the CLI sends conforms to the pinned gateway contract.

    Pass ``body=MISSING`` (the default) when the CLI sends NO body; pass the
    actual decoded body (including ``None`` for a JSON ``null`` or ``{}`` for an
    empty object) when it does. The distinction matters: ``{}``/``null``/``0``
    are bodies the gateway can reject, so they must not be conflated with "no
    body sent".

    Raises ``ConformanceError`` on any shape drift. ``query`` is accepted for a
    stable call signature (so command tests can pass captured query params) but
    not validated here: query params are nearly all GET-side and were not the
    source of the PR-#12 drifts. Validating them is a cheap follow-up if a
    query-param drift ever bites.

    This validates TOP-LEVEL request SHAPE only — it does not recurse into
    nested object/array properties, and it does not validate value types/enums.
    That is sufficient for the eight PR-#12 drifts (all top-level key/method/path
    issues); see the README "Limitations" note. The hard failures are:

    * no operation registered for ``(method, path)``  -> unregistered method/path
    * a body sent to a no-JSON-body operation          -> body where none declared
    * no body sent to an operation whose requestBody is required -> missing body
    * key in ``body`` not declared AND ``additionalProperties is False`` -> extra
    * a ``required`` property missing from ``body``    -> missing required
    """
    operation = resolve_operation(spec, method, path)
    if operation is None:
        raise ConformanceError(
            f"unregistered method/path: {method.upper()} {path} has no operation "
            f"in the pinned gateway contract (this is what a PATCH->405 or a "
            f"/runs->404 drift looks like statically)"
        )

    schema, body_required = _request_schema(spec, operation)
    body_sent = not isinstance(body, _Missing)

    # A body-less operation: sending ANY body (even {} or null) is a drift.
    if schema is None:
        if body_sent:
            shown = sorted(body) if isinstance(body, dict) else body
            raise ConformanceError(
                f"{method.upper()} {path} declares no JSON request body, but the "
                f"CLI sent: {shown!r}"
            )
        return

    # The operation declares a JSON body. If it is required, the CLI must send one.
    if not body_sent:
        if body_required:
            raise ConformanceError(
                f"{method.upper()} {path} requires a JSON request body but the CLI "
                f"sent none"
            )
        return

    schema = _merge_allof(spec, schema)

    # A present body that is NOT a JSON object, sent to an object schema, is a
    # top-level shape drift (e.g. the CLI sends a bare list/string/null where the
    # gateway expects `{...}`). We treat the schema as object-shaped when it says
    # ``type: object`` or carries any object-only facet (properties / required /
    # additionalProperties). Non-object schemas (a top-level array/string body)
    # are left to the server — they were not a PR-#12 drift class.
    if not isinstance(body, dict):
        object_shaped = (
            schema.get("type") == "object"
            or "properties" in schema
            or "required" in schema
            or "additionalProperties" in schema
        )
        if object_shaped:
            raise ConformanceError(
                f"{method.upper()} {path}: expected a JSON object body but the CLI "
                f"sent {type(body).__name__} ({body!r})"
            )
        return

    properties = schema.get("properties") or {}
    additional = schema.get("additionalProperties")

    # (2) undeclared keys, only a hard fail when extras are explicitly forbidden.
    if additional is False:
        undeclared = [k for k in body if k not in properties]
        if undeclared:
            raise ConformanceError(
                f"{method.upper()} {path}: undeclared key(s) {sorted(undeclared)} "
                f"sent to a schema with additionalProperties:false "
                f"(declared keys: {sorted(properties)}). This is the "
                f"extra='forbid' 422 the CLI used to hit."
            )

    # (3) missing required properties.
    required = schema.get("required") or []
    missing = [k for k in required if k not in body]
    if missing:
        raise ConformanceError(
            f"{method.upper()} {path}: missing required propert(y/ies) "
            f"{sorted(missing)} (sent keys: {sorted(body)})"
        )
