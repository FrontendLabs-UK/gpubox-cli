"""CLI <-> gateway contract-conformance harness.

The CLI builds request bodies for the live gateway. The gateway's request
models are Pydantic ``extra='forbid'`` — so an unknown key 422s, a missing
required field 422s, and a wrong method/path 404/405s. PR #12 fixed eight such
drifts after they shipped. This package is the systematic guard that would have
caught all eight BEFORE merge.

Two parts:

* ``conformance.py`` — ``assert_request_conforms(spec, method, path, body)``: a
  dependency-free validator that resolves the operation for ``(method, path)``
  against the PINNED gateway OpenAPI snapshot and checks request SHAPE
  (unregistered method/path, undeclared keys under ``additionalProperties:
  false``, missing required props).
* ``gateway-openapi.json`` — the pinned snapshot (the CLI's authoritative view
  of the contract). Refresh it with ``scripts/refresh-contract.sh`` whenever the
  gateway contract changes.
"""
