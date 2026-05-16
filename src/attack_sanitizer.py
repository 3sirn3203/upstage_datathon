"""
Remove known prompt-injection boilerplate from retrieved chunks.

The attack templates live in src.attacks. Placeholder values such as
<case_id>, <person>, and <date> are intentionally treated as variable text.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(r"<[A-Za-z0-9_]+>")


def sanitize_retrieval_results(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Return retrieval result copies whose chunk text has known attack messages removed."""
    sanitized_results = []
    removed_total = 0
    for result in results:
        item = result.copy()
        chunk = dict(item.get("chunk") or {})
        cleaned_text, removed_count = sanitize_attack_messages(str(chunk.get("text", "")))
        if removed_count:
            chunk["text"] = cleaned_text
            item["chunk"] = chunk
            item["removed_attack_messages"] = removed_count
            removed_total += removed_count
        sanitized_results.append(item)
    return sanitized_results, removed_total


def sanitize_attack_messages(text: str) -> tuple[str, int]:
    cleaned = str(text)
    removed_total = 0
    for pattern in attack_patterns():
        cleaned, removed_count = pattern.subn("", cleaned)
        removed_total += removed_count
    if removed_total:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, removed_total


@lru_cache(maxsize=1)
def attack_patterns() -> tuple[re.Pattern[str], ...]:
    return tuple(template_to_pattern(template) for template in attack_templates())


def attack_templates() -> list[str]:
    try:
        from . import attacks
    except ImportError:
        return []

    raw_templates = attacks.__doc__ or module_docstring_from_source(getattr(attacks, "__file__", ""))
    return [
        block.strip()
        for block in re.split(r"\n\s*\n", raw_templates.strip())
        if block.strip()
    ]


def module_docstring_from_source(path: str) -> str:
    try:
        source = Path(path).read_text(encoding="utf-8")
        return ast.get_docstring(ast.parse(source), clean=False) or ""
    except (OSError, SyntaxError):
        return ""


def template_to_pattern(template: str) -> re.Pattern[str]:
    parts = []
    idx = 0
    for match in PLACEHOLDER_RE.finditer(template):
        parts.append(literal_to_flexible_regex(template[idx : match.start()]))
        parts.append(r"[\s\S]+?")
        idx = match.end()
    parts.append(literal_to_flexible_regex(template[idx:]))
    return re.compile("".join(parts), flags=re.IGNORECASE)


def literal_to_flexible_regex(value: str) -> str:
    parts = []
    idx = 0
    while idx < len(value):
        char = value[idx]
        if char.isspace():
            while idx < len(value) and value[idx].isspace():
                idx += 1
            parts.append(r"\s+")
            continue
        if char == '"':
            while idx < len(value) and value[idx] == '"':
                idx += 1
            parts.append(r'"+')
            continue
        parts.append(re.escape(char))
        idx += 1
    return "".join(parts)
