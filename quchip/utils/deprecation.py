"""Warnings for public names scheduled for removal in quchip 0.5."""

import warnings


def warn_renamed(old: str, new: str) -> None:
    """Warn that a compatibility name will be removed in quchip 0.5."""
    warnings.warn(
        f"{old} is deprecated; use {new}. The old name will be removed in quchip 0.5.",
        DeprecationWarning,
        stacklevel=3,
    )
