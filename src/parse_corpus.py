"""
parse_corpus.py — Convert PDF corpus files into reusable text artifacts.

The parser backend is controlled by config.yaml:
    parsing.backend: pdfplumber | upstage_api

Outputs:
    parsed_corpus/<backend>/pages.jsonl
        One JSON object per parsed page.

    parsed_corpus/<backend>/text/<pdf_stem>.txt
        Human-readable text dump for quick inspection.

    parsed_corpus/upstage_api/raw/<pdf_stem>.json
        Raw Upstage API responses, when using the upstage_api backend.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from time import sleep
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pdfplumber
import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_CORPUS_DIR = Path("distribution/corpus")
DEFAULT_OUTPUT_DIR = Path("parsed_corpus")
SUPPORTED_BACKENDS = {"pdfplumber", "upstage_api"}
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


def call_upstage_document_parse(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        raise EnvironmentError("UPSTAGE_API_KEY is required for parsing.backend=upstage_api")

    import requests

    data = {
        "model": config.get("model", "document-parse"),
        "ocr": config.get("ocr", "auto"),
        "output_formats": str(config.get("output_formats", ["markdown"])),
    }
    if config.get("base64_encoding") is not None:
        data["base64_encoding"] = str(config["base64_encoding"])
    if config.get("mode"):
        data["mode"] = config["mode"]
    if config.get("coordinates") is not None:
        data["coordinates"] = str(bool(config["coordinates"])).lower()

    with path.open("rb") as file:
        response = requests.post(
            config.get("url", "https://api.upstage.ai/v1/document-digitization"),
            headers={"Authorization": f"Bearer {api_key}"},
            files={"document": file},
            data=data,
            timeout=int(config.get("timeout_seconds", 300)),
        )

    response.raise_for_status()
    return response.json()


def _extract_content(value: Any) -> str:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        for key in ("markdown", "html", "text"):
            if key in value:
                text = _extract_content(value[key])
                if text:
                    return text
    return ""


def _extract_upstage_element(item: dict[str, Any]) -> dict[str, Any] | None:
    text = (
        _extract_content(item.get("content"))
        or _extract_content(item.get("markdown"))
        or _extract_content(item.get("html"))
        or _extract_content(item.get("text"))
    )
    if not text:
        return None

    element = {
        "id": item.get("id"),
        "category": item.get("category", ""),
        "text": text,
    }
    if item.get("coordinates") is not None:
        element["coordinates"] = item["coordinates"]
    return element


def upstage_response_to_pages(source: str, response: dict[str, Any]) -> list[dict[str, Any]]:
    pages: dict[int, list[str]] = defaultdict(list)
    page_elements: dict[int, list[dict[str, Any]]] = defaultdict(list)

    elements = response.get("elements")
    if isinstance(elements, list):
        for item in elements:
            if not isinstance(item, dict):
                continue
            page = int(item.get("page") or item.get("page_number") or item.get("page_idx") or 1)
            element = _extract_upstage_element(item)
            if element:
                pages[page].append(element["text"])
                page_elements[page].append(element)

    page_items = response.get("pages")
    if isinstance(page_items, list):
        for item in page_items:
            if not isinstance(item, dict):
                continue
            page = int(item.get("page") or item.get("page_number") or item.get("page_idx") or 1)
            if page in pages:
                continue
            text = (
                _extract_content(item.get("content"))
                or _extract_content(item.get("markdown"))
                or _extract_content(item.get("html"))
                or _extract_content(item.get("text"))
            )
            if text:
                pages[page].append(text)

    if not pages:
        text = (
            _extract_content(response.get("content"))
            or _extract_content(response.get("markdown"))
            or _extract_content(response.get("html"))
            or _extract_content(response.get("text"))
        )
        if text:
            pages[1].append(text)

    return [
        {
            "source": source,
            "page": page,
            "backend": "upstage_api",
            "text": "\n\n".join(parts),
            "tables": [
                {
                    "table_index": table_index,
                    "text": element["text"],
                    "element_id": element.get("id"),
                }
                for table_index, element in enumerate(
                    [element for element in page_elements.get(page, []) if element.get("category") == "table"],
                    start=1,
                )
            ],
            "elements": page_elements.get(page, []),
        }
        for page, parts in sorted(pages.items())
    ]


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

    if jsonl_path.exists() and not should_force:
        print(f"[parse_corpus] using cached parse: {jsonl_path}")
        return jsonl_path

    pdf_paths = sorted(corpus_path.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {corpus_path}")

    text_dir = backend_output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    total_pages = 0

    raw_dir = backend_output_dir / "raw"
    if backend == "upstage_api":
        raw_dir.mkdir(parents=True, exist_ok=True)

    tmp_jsonl_path = jsonl_path.with_suffix(".jsonl.tmp")
    if tmp_jsonl_path.exists():
        tmp_jsonl_path.unlink()

    try:
        with tmp_jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for pdf_path in pdf_paths:
                if backend == "pdfplumber":
                    backend_config = parsing_config.get("pdfplumber", {})
                    records = list(
                        parse_pdf_with_pdfplumber(
                            pdf_path,
                            extract_tables=backend_config.get("extract_tables", True),
                        )
                    )
                else:
                    response = call_upstage_document_parse(pdf_path, parsing_config.get("upstage_api", {}))
                    (raw_dir / f"{pdf_path.stem}.json").write_text(
                        json.dumps(response, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    records = upstage_response_to_pages(pdf_path.name, response)
                    sleep(2)

                total_pages += len(records)
                for record in records:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                write_text_dump(records, text_dir / f"{pdf_path.stem}.txt")
                print(f"[parsed:{backend}] {pdf_path.name}: {len(records)} pages")
    except Exception:
        if tmp_jsonl_path.exists():
            tmp_jsonl_path.unlink()
        raise

    tmp_jsonl_path.replace(jsonl_path)
    print(f"[done] wrote {total_pages} pages to {jsonl_path}")
    print(f"[done] wrote text dumps to {text_dir}")
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
