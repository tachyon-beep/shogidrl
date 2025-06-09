"""
Legacy evaluation components for backward compatibility.

This module contains the original evaluation files moved here during the refactor
to maintain backward compatibility while the new system is being developed.
"""

# Import only EloRegistry here to avoid heavy dependencies during testing
try:  # pragma: no cover - optional import
    from .elo_registry import EloRegistry
    __all__ = ["EloRegistry"]
except Exception:  # noqa: BLE001 - graceful fallback if file missing
    __all__ = []
