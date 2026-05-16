"""
parse_corpus.py — Convert PDF corpus files into reusable text artifacts.

The parser uses pdfplumber only.

Outputs:
    parsed_corpus/pdfplumber/pages.jsonl
        One JSON object per parsed page.

    parsed_corpus/pdfplumber/text/<pdf_stem>.txt
        Human-readable text dump for quick inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pdfplumber
import yaml

try:
    from .logging_utils import log_block
except ImportError:
    from logging_utils import log_block


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_CORPUS_DIR = Path("distribution/corpus")
DEFAULT_OUTPUT_DIR = Path("parsed_corpus")
SUPPORTED_BACKENDS = {"pdfplumber"}
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    value = CONTROL_CHARS.sub("", str(value))
    return re.sub(r"\s+", " ", value).strip()


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = CONTROL_CHARS.sub("", value)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def table_to_text(table: list[list[object]]) -> str:
    rows = [[clean_cell(cell) for cell in row] for row in table if row]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""

    header = rows[0]
    body = rows[1:]
    rendered = []

    if body and any(header):
        for row in body:
            pairs = []
            for idx, cell in enumerate(row):
                key = header[idx] if idx < len(header) and header[idx] else f"column_{idx + 1}"
                if cell:
                    pairs.append(f"{key}: {cell}")
            if pairs:
                rendered.append(" | ".join(pairs))
    else:
        rendered = [" | ".join(cell for cell in row if cell) for row in rows]

    return "\n".join(rendered)


def parse_pdf_with_pdfplumber(path: Path, *, extract_tables: bool = True) -> Iterable[dict[str, Any]]:
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_text = clean_text(page.extract_text())
            tables = []

            if extract_tables:
                for table_index, table in enumerate(page.extract_tables(), start=1):
                    table_text = table_to_text(table)
                    if table_text:
                        tables.append(
                            {
                                "table_index": table_index,
                                "text": table_text,
                                "rows": [[clean_cell(cell) for cell in row] for row in table if row],
                            }
                        )

            yield {
                "source": path.name,
                "page": page_index,
                "backend": "pdfplumber",
                "text": page_text,
                "tables": tables,
            }


def write_text_dump(records: list[dict[str, Any]], output_path: Path) -> None:
    parts = []
    for record in records:
        parts.append(f"===== {record['source']} / page {record['page']} =====")
        if record["text"]:
            parts.append(record["text"])
        for table in record["tables"]:
            parts.append(f"\n[Table {table['table_index']}]\n{table['text']}")
        parts.append("")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_page_records(records: list[dict[str, Any]], output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)


def load_page_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_single_pdf(
    pdf_path: Path,
    *,
    parsing_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    backend_config = parsing_config.get("pdfplumber", {})
    records = list(
        parse_pdf_with_pdfplumber(
            pdf_path,
            extract_tables=backend_config.get("extract_tables", True),
        )
    )
    return records, f"{len(records)} pages"


def seed_page_cache_from_combined_jsonl(jsonl_path: Path, page_cache_dir: Path) -> int:
    if not jsonl_path.exists() or any(page_cache_dir.glob("*.jsonl")):
        return 0

    records_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_page_records(jsonl_path):
        source = str(record.get("source") or "")
        if source:
            records_by_source[source].append(record)

    for source, records in records_by_source.items():
        write_page_records(records, page_cache_dir / f"{Path(source).stem}.jsonl")
    return len(records_by_source)


def combined_cache_covers_corpus(jsonl_path: Path, pdf_paths: list[Path]) -> bool:
    if not jsonl_path.exists():
        return False

    expected_sources = {pdf_path.name for pdf_path in pdf_paths}
    cached_sources = {
        str(record.get("source") or "")
        for record in load_page_records(jsonl_path)
        if record.get("source")
    }
    return expected_sources <= cached_sources


def parse_corpus(
    corpus_dir: str | Path | None = None,
    *,
    option: str | None = None,
    output_dir: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    force: bool | None = None,
) -> Path:
    config = load_config(config_path)
    parsing_config = config.get("parsing", {})

    backend = option or parsing_config.get("backend", "pdfplumber")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported parser backend: {backend}. Use one of {sorted(SUPPORTED_BACKENDS)}")

    corpus_path = Path(corpus_dir or config.get("corpus", {}).get("dir", DEFAULT_CORPUS_DIR))
    base_output_dir = Path(output_dir or parsing_config.get("output_dir", DEFAULT_OUTPUT_DIR))
    backend_output_dir = base_output_dir / backend
    jsonl_path = backend_output_dir / "pages.jsonl"
    should_force = parsing_config.get("force", False) if force is None else force

    pdf_paths = sorted(corpus_path.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {corpus_path}")

    if jsonl_path.exists() and not should_force and combined_cache_covers_corpus(jsonl_path, pdf_paths):
        log_block(
            f"Parse:{backend}",
            "Cache hit",
            f"Using cached parse: {jsonl_path}",
        )
        return jsonl_path

    if jsonl_path.exists() and not should_force:
        log_block(
            f"Parse:{backend}",
            "Cache incomplete",
            f"Existing combined cache does not cover current corpus. Resuming with per-document cache: {jsonl_path}",
        )

    text_dir = backend_output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    page_cache_dir = backend_output_dir / "pages"
    page_cache_dir.mkdir(parents=True, exist_ok=True)
    seeded_cache_count = seed_page_cache_from_combined_jsonl(jsonl_path, page_cache_dir) if not should_force else 0
    total_pages = 0

    tmp_jsonl_path = jsonl_path.with_suffix(".jsonl.tmp")
    if tmp_jsonl_path.exists():
        tmp_jsonl_path.unlink()

    parsed_summaries = []
    if seeded_cache_count:
        parsed_summaries.append(f"- seeded per-document cache from existing pages.jsonl: {seeded_cache_count} files")
    try:
        for index, pdf_path in enumerate(pdf_paths, start=1):
            page_cache_path = page_cache_dir / f"{pdf_path.stem}.jsonl"
            if page_cache_path.exists() and not should_force:
                records = load_page_records(page_cache_path)
                parse_summary = "resumed from per-document cache"
            else:
                records, parse_summary = parse_single_pdf(
                    pdf_path,
                    parsing_config=parsing_config,
                )
                write_page_records(records, page_cache_path)
                write_text_dump(records, text_dir / f"{pdf_path.stem}.txt")

            total_pages += len(records)
            parsed_summaries.append(f"- {pdf_path.name}: {len(records)} pages ({parse_summary})")
            print(
                f"[parse:{backend}] {index}/{len(pdf_paths)} {pdf_path.name}: "
                f"{len(records)} pages ({parse_summary})",
                flush=True,
            )

        with tmp_jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for pdf_path in pdf_paths:
                page_cache_path = page_cache_dir / f"{pdf_path.stem}.jsonl"
                for record in load_page_records(page_cache_path):
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        if tmp_jsonl_path.exists():
            tmp_jsonl_path.unlink()
        raise

    tmp_jsonl_path.replace(jsonl_path)
    log_block(
        f"Parse:{backend}",
        "Completed",
        "\n".join(
            [
                *parsed_summaries,
                f"Total pages: {total_pages}",
                f"Pages JSONL: {jsonl_path}",
                f"Text dumps: {text_dir}",
            ]
        ),
    )
    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDF corpus files.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--option", choices=sorted(SUPPORTED_BACKENDS), help="Parser backend override.")
    parser.add_argument("--force", action="store_true", help="Ignore cached pages.jsonl and parse again.")
    args = parser.parse_args()

    parse_corpus(
        corpus_dir=args.corpus_dir,
        option=args.option,
        output_dir=args.output_dir,
        config_path=args.config,
        force=args.force or None,
    )


if __name__ == "__main__":
    main()
