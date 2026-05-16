"""
index_corpus.py — Build dense retrieval indexes from corpus chunks.

Input:
    parsed_corpus/<backend>/chunks.jsonl

Outputs:
    parsed_corpus/<backend>/dense.faiss
        FAISS IndexFlatIP over normalized bge-large-en-v1.5 passage embeddings.

    parsed_corpus/<backend>/dense_metadata.jsonl
        One metadata row per FAISS vector, preserving chunk fields except text is kept
        as-is for retrieval context construction.

    parsed_corpus/<backend>/dense_embeddings.npy
        Normalized float32 embedding matrix. Useful for inspection/fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore
except ImportError:
    from parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore


@dataclass
class DenseIndexConfig:
    passage_model: str = "BAAI/bge-large-en-v1.5"
    query_model: str = "BAAI/bge-large-en-v1.5"  # 동일 모델, 쿼리 시 사용
    dimension: int = 1024
    batch_size: int = 64
    force: bool = False
    faiss_filename: str = "dense.faiss"
    metadata_filename: str = "dense_metadata.jsonl"
    embeddings_filename: str = "dense_embeddings.npy"


def config_from_dict(config: dict[str, Any]) -> DenseIndexConfig:
    dense = config.get("indexing", {}).get("dense", {})
    return DenseIndexConfig(
        passage_model=str(dense.get("passage_model", "BAAI/bge-large-en-v1.5")),
        query_model=str(dense.get("query_model", "BAAI/bge-large-en-v1.5")),
        dimension=int(dense.get("dimension", 1024)),
        batch_size=int(dense.get("batch_size", 64)),
        force=bool(dense.get("force", False)),
        faiss_filename=str(dense.get("faiss_filename", "dense.faiss")),
        metadata_filename=str(dense.get("metadata_filename", "dense_metadata.jsonl")),
        embeddings_filename=str(dense.get("embeddings_filename", "dense_embeddings.npy")),
    )


def default_chunks_path(config: dict[str, Any]) -> Path:
    parsing = config.get("parsing", {})
    chunking = config.get("chunking", {})
    backend = parsing.get("backend", "pdfplumber")
    output_dir = Path(parsing.get("output_dir", "parsed_corpus"))
    chunk_filename = chunking.get("output_filename", "chunks.jsonl")
    return output_dir / backend / chunk_filename


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    for chunk in iter_jsonl(path):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        chunks.append(chunk)
    return chunks


def embed_texts(texts: list[str], model_name: str, config: DenseIndexConfig) -> np.ndarray:
    """sentence-transformers로 로컬 임베딩 (bge-large-en-v1.5)."""
    if not texts:
        return np.empty((0, config.dimension), dtype=np.float32)

    print(f"[index_corpus] loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[index_corpus] encoding {len(texts)} texts (batch_size={config.batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=config.batch_size,
        normalize_embeddings=True,   # bge는 normalize 권장
        show_progress_bar=True,
    )

    return embeddings.astype(np.float32)


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    """embed_texts에서 이미 normalize했지만, 외부 호출 시 안전망으로 유지."""
    normalized = matrix.astype(np.float32, copy=True)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized /= norms
    return normalized


def write_metadata(path: Path, chunks: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for vector_id, chunk in enumerate(chunks):
            record = dict(chunk)
            record["vector_id"] = vector_id
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dense_index(
    chunks_path: str | Path,
    *,
    config: DenseIndexConfig | None = None,
    output_dir: str | Path | None = None,
    force: bool | None = None,
) -> dict[str, Path | int]:
    chunks_path = Path(chunks_path)
    dense_config = config or DenseIndexConfig()
    should_force = dense_config.force if force is None else force
    target_dir = Path(output_dir) if output_dir else chunks_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = target_dir / dense_config.faiss_filename
    metadata_path = target_dir / dense_config.metadata_filename
    embeddings_path = target_dir / dense_config.embeddings_filename

    if (
        faiss_path.exists()
        and metadata_path.exists()
        and embeddings_path.exists()
        and count_jsonl(chunks_path) == count_jsonl(metadata_path)
        and not should_force
    ):
        print(f"[index_corpus] using cached dense index: {faiss_path}")
        return {
            "faiss_path": faiss_path,
            "metadata_path": metadata_path,
            "embeddings_path": embeddings_path,
            "num_vectors": count_jsonl(metadata_path),
        }

    chunks = load_chunks(chunks_path)
    texts = [chunk["text"] for chunk in chunks]

    print(f"[index_corpus] embedding {len(texts)} chunks with {dense_config.passage_model}")
    embeddings = embed_texts(texts, dense_config.passage_model, dense_config)
    # sentence-transformers에서 normalize_embeddings=True로 이미 정규화됨
    # 혹시 모를 수치 오차 보정
    embeddings = normalize_embeddings(embeddings)

    index = faiss.IndexFlatIP(dense_config.dimension)
    index.add(embeddings)

    faiss.write_index(index, str(faiss_path))
    np.save(embeddings_path, embeddings)
    write_metadata(metadata_path, chunks)

    print(f"[index_corpus] wrote FAISS index: {faiss_path} ({index.ntotal} vectors)")
    print(f"[index_corpus] wrote metadata: {metadata_path}")
    print(f"[index_corpus] wrote embeddings: {embeddings_path}")
    return {
        "faiss_path": faiss_path,
        "metadata_path": metadata_path,
        "embeddings_path": embeddings_path,
        "num_vectors": int(index.ntotal),
    }


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS dense index from chunks.jsonl.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--chunks", type=Path, help="Path to chunks.jsonl. Defaults to config backend.")
    parser.add_argument("--output-dir", type=Path, help="Directory for dense index artifacts.")
    parser.add_argument("--force", action="store_true", help="Ignore cached dense index and rebuild.")
    args = parser.parse_args()

    raw_config = load_config(args.config)
    chunks_path = args.chunks or default_chunks_path(raw_config)
    build_dense_index(
        chunks_path,
        config=config_from_dict(raw_config),
        output_dir=args.output_dir,
        force=args.force or None,
    )


if __name__ == "__main__":
    main()