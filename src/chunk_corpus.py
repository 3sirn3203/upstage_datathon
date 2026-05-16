"""
chunk_corpus.py — Build naive fixed-size retrieval chunks from parsed pages.

Input:
    parsed_corpus/pdfplumber/pages.jsonl

Output:
    parsed_corpus/pdfplumber/chunks.jsonl

Strategy:
    - Use the parsed page text plus any pdfplumber table text.
    - Split by fixed character windows.
    - Control window size and overlap via config.yaml.
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
    from .logging_utils import log_block  # type: ignore
    from .parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore
except ImportError:
    from logging_utils import log_block  # type: ignore
    from parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore


@dataclass
class ChunkingConfig:
    max_chunk_chars: int = 1200
    overlap_chars: int = 400
    output_filename: str = "chunks.jsonl"


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).splitlines()]
    normalized = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def table_rows_to_text(rows: list[list[Any]]) -> str:
    rendered_rows = []
    for row in rows:
        cells = [str(cell or "").strip() for cell in row]
        cells = [cell for cell in cells if cell]
        if cells:
            rendered_rows.append(" | ".join(cells))
    return "\n".join(rendered_rows)


def record_text(record: dict[str, Any]) -> str:
    parts = []
    page_text = normalize_text(record.get("text", ""))
    if page_text:
        parts.append(page_text)

    for table in record.get("tables", []):
        if not isinstance(table, dict):
            continue
        table_text = normalize_text(table.get("text") or table_rows_to_text(table.get("rows", [])))
        if not table_text:
            continue
        table_index = table.get("table_index", "")
        label = f"[Table {table_index}]" if table_index else "[Table]"
        parts.append(f"{label}\n{table_text}")

    return normalize_text("\n\n".join(parts))


def split_fixed_windows(text: str, max_chars: int, overlap_chars: int) -> list[tuple[str, int, int]]:
    text = normalize_text(text)
    if not text:
        return []
    if max_chars <= 0:
        raise ValueError("chunking.max_chunk_chars must be greater than 0")
    if overlap_chars < 0:
        raise ValueError("chunking.overlap_chars must be greater than or equal to 0")
    if overlap_chars >= max_chars:
        raise ValueError("chunking.overlap_chars must be smaller than chunking.max_chunk_chars")

    chunks: list[tuple[str, int, int]] = []
    start = 0
    step = max_chars - overlap_chars
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, start, end))
        if end >= text_length:
            break
        start += step

    return chunks


def make_chunk(
    *,
    source: str,
    page: int,
    backend: str,
    sequence: int,
    text: str,
    start_char: int,
    end_char: int,
) -> dict[str, Any]:
    return {
        "chunk_id": f"{Path(source).stem}:p{page}:chunk:{sequence:04d}",
        "source": source,
        "page": page,
        "backend": backend,
        "kind": "chunk",
        "sequence": sequence,
        "section": f"page {page} chunk {sequence}",
        "text": text,
        "start_char": start_char,
        "end_char": end_char,
    }


def build_chunks_for_record(record: dict[str, Any], config: ChunkingConfig) -> list[dict[str, Any]]:
    source = str(record.get("source", ""))
    page = int(record.get("page") or 0)
    backend = str(record.get("backend", ""))
    text = record_text(record)

    chunks = []
    for sequence, (chunk_text, start_char, end_char) in enumerate(
        split_fixed_windows(text, config.max_chunk_chars, config.overlap_chars),
        start=1,
    ):
        chunks.append(
            make_chunk(
                source=source,
                page=page,
                backend=backend,
                sequence=sequence,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
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
    with output.open("w", encoding="utf-8") as file:
        for record in load_pages(pages_path):
            for chunk in build_chunks_for_record(record, chunking_config):
                chunk_count += 1
                chunk["global_sequence"] = chunk_count
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log_block(
        "Index Build",
        "Chunk corpus",
        (
            f"Chunks: {chunk_count}\n"
            f"Max chars: {chunking_config.max_chunk_chars}\n"
            f"Overlap chars: {chunking_config.overlap_chars}\n"
            f"Output: {output}"
        ),
    )
    return output


def config_from_dict(config: dict[str, Any]) -> ChunkingConfig:
    chunking = config.get("chunking", {})
    return ChunkingConfig(
        max_chunk_chars=int(chunking.get("max_chunk_chars", 1200)),
        overlap_chars=int(chunking.get("overlap_chars", 400)),
        output_filename=str(chunking.get("output_filename", "chunks.jsonl")),
    )


def default_pages_path(config: dict[str, Any]) -> Path:
    parsing = config.get("parsing", {})
    backend = parsing.get("backend", "pdfplumber")
    output_dir = Path(parsing.get("output_dir", "parsed_corpus"))
    return output_dir / backend / "pages.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk parsed corpus pages with fixed-size windows.")
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
