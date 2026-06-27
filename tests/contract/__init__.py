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
* ``gateway-openapi.json`` — the pinned snapshot, kept BYTE-IDENTICAL to the
  gateway's canonical published spec (gpubox-gateway commits
  ``openapi/openapi.json`` and fails its own CI if that file drifts from its
  code). Refresh it with ``scripts/refresh-contract.sh`` (pulls the canonical
  spec by default); the nightly ``contract-drift`` workflow re-pulls and fails if
  the pin falls out of sync — so the gateway moving underneath the pin is caught
  automatically, not only on a manual refresh.
"""
