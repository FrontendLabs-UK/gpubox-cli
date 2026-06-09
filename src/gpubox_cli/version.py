"""Single source of truth for the CLI version + User-Agent string."""

from __future__ import annotations

__version__ = "0.1.1"

#: Sent on every HTTP request to the GPUBox API. Don't change the shape — the
#: gateway parses it for usage analytics.
USER_AGENT = f"gpubox-cli/{__version__} (+https://gpubox.ai)"
