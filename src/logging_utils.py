"""Shared terminal logging helpers for pipeline stages."""

from __future__ import annotations


SEPARATOR = "=" * 20


def log_block(stage: str, name: str, log: str = "", *, leading_newlines: int = 1) -> None:
    """Print a consistently separated terminal log block."""
    print(f"{chr(10) * leading_newlines}[{stage}] {name}")
    if log:
        print(str(log).rstrip())
    print(f"\n{SEPARATOR}")
