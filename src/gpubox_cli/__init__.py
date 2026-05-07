"""GPUBox CLI — customer-facing client for the GPUBox AI inference platform.

Public entry point: :mod:`gpubox_cli.main`. Most modules here are internal.
"""

from gpubox_cli.version import USER_AGENT, __version__

__all__ = ["__version__", "USER_AGENT"]
