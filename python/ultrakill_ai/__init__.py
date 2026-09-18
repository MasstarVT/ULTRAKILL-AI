"""The ULTRAKILL AI package.

**Importing this package is deliberately cheap.** It used to do `from ultrakill_ai.env import ...` at the
top, so *any* submodule import -- `ultrakill_ai.times` for the leaderboard, `ultrakill_ai.windows` for a
monitor rectangle -- pulled in `env` and with it gymnasium and numpy, and numpy costs 785 MB of committed
private bytes on this box (see `procmem`). That is why `dashboard.py`, `campaign_driver.py` and
`post_times.py` each sat at 0.75 GB of commit for 0.04 GB of working set.

The public names are still importable (`from ultrakill_ai import UltrakillEnv` works), they are just
resolved on first use through PEP 562's module `__getattr__`, so nothing is paid by a process that never
touches them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY = {
    "EnvConfig": "ultrakill_ai.env",
    "UltrakillEnv": "ultrakill_ai.env",
    "RECOVERABLE": "ultrakill_ai.protocol",
    "BridgeClient": "ultrakill_ai.protocol",
    "BridgeClosed": "ultrakill_ai.protocol",
    "BridgeError": "ultrakill_ai.protocol",
    "BridgeSceneUnknown": "ultrakill_ai.protocol",
    "BridgeTimeout": "ultrakill_ai.protocol",
}

__all__ = sorted(_LAZY)

if TYPE_CHECKING:  # for type checkers and editors only; never executed
    from ultrakill_ai.env import EnvConfig, UltrakillEnv
    from ultrakill_ai.protocol import (
        RECOVERABLE,
        BridgeClient,
        BridgeClosed,
        BridgeError,
        BridgeSceneUnknown,
        BridgeTimeout,
    )


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # resolved once; later reads never reach here
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
