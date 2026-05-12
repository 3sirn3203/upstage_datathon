"""
chunk_corpus.py — Build retrieval chunks from parsed corpus pages.

Input:
    parsed_corpus/<backend>/pages.jsonl

Output:
    parsed_corpus/<backend>/chunks.jsonl

Strategy:
    - Split Markdown text by heading sections (#, ##, ###, ...).
    - Keep heading path metadata on every chunk.
    - Keep tables inside paragraph/section chunks.
    - Emit additional table chunks for Markdown tables and pdfplumber tables.
    - If a section is too long, split with overlap while preserving block boundaries
      when possible.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore
except ImportError:
    from parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


@dataclass
class ChunkingConfig:
    max_chunk_chars: int = 1200
    overlap_chars: int = 200
    min_chunk_chars: int = 80
    include_table_chunks: bool = True
    output_filename: str = "chunks.jsonl"


@dataclass
class Section:
    heading_path: list[str]
    text: str


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    normalized = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def heading_text(line: str) -> str:
    match = HEADING_RE.match(line)
    return match.group(2).strip() if match else ""


def split_markdown_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal current_lines, current_path
        body = normalize_text("\n".join(current_lines))
        if body:
            sections.append(Section(heading_path=current_path.copy(), text=body))
        current_lines = []

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence

        match = HEADING_RE.match(line) if not in_fence else None
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_path = heading_stack.copy()
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()
    return sections


def split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_table = False

    for line in text.splitlines():
        is_table_row = bool(TABLE_ROW_RE.match(line))
        if is_table_row:
            if current and not in_table:
                blocks.append(normalize_text("\n".join(current)))
                current = []
            current.append(line)
            in_table = True
            continue

        if in_table:
            blocks.append(normalize_text("\n".join(current)))
            current = []
            in_table = False

        if line.strip():
            current.append(line)
        elif current:
            blocks.append(normalize_text("\n".join(current)))
            current = []

    if current:
        blocks.append(normalize_text("\n".join(current)))

    return [block for block in blocks if block]


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0
    overlap = min(overlap_chars, max_chars // 3)

    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("。", start, end),
            )
            if boundary > start + max_chars // 2:
                end = boundary + 1

        chunk = normalize_text(text[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def split_section_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    blocks = split_blocks(text)
    if not blocks:
        return split_long_text(text, max_chars, overlap_chars)

    chunks: list[str] = []
    current: list[str] = []

    for block in blocks:
        candidate = normalize_text("\n\n".join(current + [block]))
        if len(candidate) <= max_chars:
            current.append(block)
            continue

        if current:
            chunks.append(normalize_text("\n\n".join(current)))

        if len(block) > max_chars:
            chunks.extend(split_long_text(block, max_chars, overlap_chars))
            current = []
        else:
            overlap_seed = build_overlap_seed(chunks[-1] if chunks else "", overlap_chars)
            current = [overlap_seed, block] if overlap_seed else [block]
            if len(normalize_text("\n\n".join(current))) > max_chars:
                current = [block]

    if current:
        chunks.append(normalize_text("\n\n".join(current)))

    return chunks


def build_overlap_seed(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or not text:
        return ""
    seed = text[-overlap_chars:]
    boundary = min(
        [idx for idx in (seed.find("\n\n"), seed.find("\n"), seed.find(". ")) if idx >= 0],
        default=-1,
    )
    if boundary >= 0 and boundary + 1 < len(seed):
        seed = seed[boundary + 1 :]
    return normalize_text(seed)


def merge_small_sections(sections: list[Section], min_chars: int) -> list[Section]:
    if min_chars <= 0:
        return sections

    merged: list[Section] = []
    for section in sections:
        if merged and len(section.text) < min_chars:
            previous = merged[-1]
            previous.text = normalize_text(f"{previous.text}\n\n{section.text}")
            continue
        merged.append(section)
    return merged


def extract_markdown_tables(text: str) -> list[str]:
    tables: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        if TABLE_ROW_RE.match(line):
            current.append(line)
            continue
        if current:
            maybe_table = normalize_text("\n".join(current))
            if is_markdown_table(maybe_table):
                tables.append(maybe_table)
            current = []

    if current:
        maybe_table = normalize_text("\n".join(current))
        if is_markdown_table(maybe_table):
            tables.append(maybe_table)

    return tables


def is_markdown_table(text: str) -> bool:
    rows = [line for line in text.splitlines() if TABLE_ROW_RE.match(line)]
    if len(rows) < 2:
        return False
    return any(TABLE_SEPARATOR_RE.match(row) for row in rows[1:3])


def make_chunk(
    *,
    source: str,
    page: int,
    backend: str,
    kind: str,
    sequence: int,
    text: str,
    heading_path: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "chunk_id": f"{Path(source).stem}:p{page}:{kind}:{sequence:04d}",
        "source": source,
        "page": page,
        "backend": backend,
        "kind": kind,
        "sequence": sequence,
        "heading_path": heading_path,
        "section": " > ".join(heading_path),
        "text": text,
    }
    if extra:
        payload.update(extra)
    return payload


def table_rows_to_markdown(rows: list[list[Any]]) -> str:
    clean_rows = [[str(cell or "").strip() for cell in row] for row in rows if row]
    clean_rows = [row for row in clean_rows if any(row)]
    if not clean_rows:
        return ""

    width = max(len(row) for row in clean_rows)
    padded = [row + [""] * (width - len(row)) for row in clean_rows]
    header = padded[0]
    body = padded[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def build_chunks_for_record(record: dict[str, Any], config: ChunkingConfig) -> list[dict[str, Any]]:
    return build_chunks_for_record_with_context(record, config, initial_heading_path=[])


def build_chunks_for_record_with_context(
    record: dict[str, Any],
    config: ChunkingConfig,
    *,
    initial_heading_path: list[str],
) -> list[dict[str, Any]]:
    source = record["source"]
    page = int(record["page"])
    backend = record.get("backend", "")
    text = normalize_text(record.get("text", ""))
    chunks: list[dict[str, Any]] = []
    sequence = 0

    sections = split_markdown_sections(text) if text else []
    if not sections and text:
        sections = [Section(heading_path=[], text=text)]
    sections = merge_small_sections(sections, config.min_chunk_chars)

    for section in sections:
        effective_heading_path = section.heading_path or initial_heading_path
        for part in split_section_text(section.text, config.max_chunk_chars, config.overlap_chars):
            sequence += 1
            chunks.append(
                make_chunk(
                    source=source,
                    page=page,
                    backend=backend,
                    kind="section",
                    sequence=sequence,
                    text=part,
                    heading_path=effective_heading_path,
                )
            )

        if config.include_table_chunks:
            for table_index, table_text in enumerate(extract_markdown_tables(section.text), start=1):
                sequence += 1
                chunks.append(
                    make_chunk(
                        source=source,
                        page=page,
                        backend=backend,
                        kind="table",
                        sequence=sequence,
                        text=table_text,
                        heading_path=effective_heading_path,
                        extra={"table_index": table_index},
                    )
                )

    if config.include_table_chunks:
        for table in record.get("tables", []):
            table_text = table.get("text") or table_rows_to_markdown(table.get("rows", []))
            table_text = normalize_text(table_text)
            if not table_text:
                continue
            sequence += 1
            chunks.append(
                make_chunk(
                    source=source,
                    page=page,
                    backend=backend,
                    kind="table",
                    sequence=sequence,
                    text=table_text,
                    heading_path=initial_heading_path,
                    extra={"table_index": table.get("table_index")},
                )
            )

    return chunks


def load_pages(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunk_corpus(
    pages_path: str | Path,
    *,
    output_path: str | Path | None = None,
    config: ChunkingConfig | None = None,
) -> Path:
    pages_path = Path(pages_path)
    chunking_config = config or ChunkingConfig()
    output = Path(output_path) if output_path else pages_path.with_name(chunking_config.output_filename)

    chunk_count = 0
    last_heading_by_source: dict[str, list[str]] = {}
    with output.open("w", encoding="utf-8") as file:
        for record in load_pages(pages_path):
            source = record.get("source", "")
            chunks = build_chunks_for_record_with_context(
                record,
                chunking_config,
                initial_heading_path=last_heading_by_source.get(source, []),
            )
            for chunk in chunks:
                chunk_count += 1
                chunk["global_sequence"] = chunk_count
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                if chunk.get("heading_path"):
                    last_heading_by_source[source] = chunk["heading_path"]

    print(f"[chunk_corpus] wrote {chunk_count} chunks to {output}")
    return output


def config_from_dict(config: dict[str, Any]) -> ChunkingConfig:
    chunking = config.get("chunking", {})
    return ChunkingConfig(
        max_chunk_chars=int(chunking.get("max_chunk_chars", 1200)),
        overlap_chars=int(chunking.get("overlap_chars", 200)),
        min_chunk_chars=int(chunking.get("min_chunk_chars", 80)),
        include_table_chunks=bool(chunking.get("include_table_chunks", True)),
        output_filename=str(chunking.get("output_filename", "chunks.jsonl")),
    )


def default_pages_path(config: dict[str, Any]) -> Path:
    parsing = config.get("parsing", {})
    backend = parsing.get("backend", "pdfplumber")
    output_dir = Path(parsing.get("output_dir", "parsed_corpus"))
    return output_dir / backend / "pages.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk parsed corpus pages.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pages", type=Path, help="Path to pages.jsonl. Defaults to config backend.")
    parser.add_argument("--output", type=Path, help="Path to write chunks.jsonl.")
    args = parser.parse_args()

    raw_config = load_config(args.config)
    pages_path = args.pages or default_pages_path(raw_config)
    chunk_corpus(
        pages_path,
        output_path=args.output,
        config=config_from_dict(raw_config),
    )


if __name__ == "__main__":
    main()
