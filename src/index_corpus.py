"""
index_corpus.py — Build dense retrieval indexes from corpus chunks.

Input:
    parsed_corpus/<backend>/chunks.jsonl

Outputs:
    parsed_corpus/<backend>/dense.faiss
        FAISS IndexFlatIP over normalized local sentence-transformers embeddings.

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

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore
    from .logging_utils import log_block  # type: ignore
except ImportError:
    from parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore
    from logging_utils import log_block  # type: ignore


@dataclass
class DenseIndexConfig:
    enabled: bool = True
    model_name: str = "BAAI/bge-large-en-v1.5"
    dimension: int = 1024
    batch_size: int = 32
    device: str = "auto"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    normalize_embeddings: bool = True
    show_progress_bar: bool = True
    max_seq_length: int = 512
    force: bool = False
    faiss_filename: str = "dense.faiss"
    metadata_filename: str = "dense_metadata.jsonl"
    embeddings_filename: str = "dense_embeddings.npy"
    manifest_filename: str = "dense_manifest.json"


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def config_from_dict(config: dict[str, Any]) -> DenseIndexConfig:
    dense = config.get("indexing", {}).get("dense", {})
    return DenseIndexConfig(
        enabled=bool(dense.get("enabled", True)),
        model_name=str(dense.get("model_name", "BAAI/bge-large-en-v1.5")),
        dimension=int(dense.get("dimension", 1024)),
        batch_size=int(dense.get("batch_size", 32)),
        device=str(dense.get("device", "auto")),
        query_instruction=str(
            dense.get("query_instruction", "Represent this sentence for searching relevant passages: ")
        ),
        normalize_embeddings=bool(dense.get("normalize_embeddings", True)),
        show_progress_bar=bool(dense.get("show_progress_bar", True)),
        max_seq_length=int(dense.get("max_seq_length", 512)),
        force=bool(dense.get("force", False)),
        faiss_filename=str(dense.get("faiss_filename", "dense.faiss")),
        metadata_filename=str(dense.get("metadata_filename", "dense_metadata.jsonl")),
        embeddings_filename=str(dense.get("embeddings_filename", "dense_embeddings.npy")),
        manifest_filename=str(dense.get("manifest_filename", "dense_manifest.json")),
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


def embed_passages(texts: list[str], config: DenseIndexConfig) -> np.ndarray:
    return embed_texts(texts, config)


def embed_queries(queries: list[str], config: DenseIndexConfig) -> np.ndarray:
    instruction = config.query_instruction
    texts = [f"{instruction}{query}" if instruction else query for query in queries]
    return embed_texts(texts, config)


def embed_texts(texts: list[str], config: DenseIndexConfig) -> np.ndarray:
    if not texts:
        return np.empty((0, config.dimension), dtype=np.float32)

    model = load_embedding_model(config)
    matrix = model.encode(
        texts,
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=config.show_progress_bar and len(texts) > config.batch_size,
    ).astype(np.float32, copy=False)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise RuntimeError(f"Expected a 2D embedding matrix, got shape {matrix.shape}")
    if config.dimension > 0 and matrix.shape[1] != config.dimension:
        raise RuntimeError(
            f"Expected embedding dimension {config.dimension} for {config.model_name}, got {matrix.shape[1]}"
        )
    return matrix


def load_embedding_model(config: DenseIndexConfig) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Local dense retrieval requires sentence-transformers. "
            "Install project requirements before building or querying the dense index."
        ) from exc

    device = resolved_device(config.device)
    cache_key = (config.model_name, device or "auto", str(config.max_seq_length))
    if cache_key not in _MODEL_CACHE:
        kwargs = {"device": device} if device else {}
        model = SentenceTransformer(config.model_name, **kwargs)
        if config.max_seq_length > 0:
            model.max_seq_length = config.max_seq_length
        _MODEL_CACHE[cache_key] = model
    return _MODEL_CACHE[cache_key]


def resolved_device(device: str) -> str | None:
    value = str(device or "").strip()
    if not value or value.lower() == "auto":
        return None
    return value


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
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


def write_manifest(path: Path, chunks_path: Path, config: DenseIndexConfig, vector_count: int, dimension: int) -> None:
    manifest = {
        "embedding_backend": "sentence-transformers",
        "model_name": config.model_name,
        "dimension": dimension,
        "normalize_embeddings": config.normalize_embeddings,
        "query_instruction": config.query_instruction,
        "max_seq_length": config.max_seq_length,
        "chunks_path": str(chunks_path),
        "chunk_count": count_jsonl(chunks_path),
        "vector_count": vector_count,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cache_matches(path: Path, chunks_path: Path, config: DenseIndexConfig) -> bool:
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest_dimension = int(manifest.get("dimension", 0))
        manifest_max_seq_length = int(manifest.get("max_seq_length", 0))
        manifest_chunk_count = int(manifest.get("chunk_count", -1))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False

    return (
        manifest.get("embedding_backend") == "sentence-transformers"
        and manifest.get("model_name") == config.model_name
        and manifest_dimension == config.dimension
        and bool(manifest.get("normalize_embeddings")) == config.normalize_embeddings
        and str(manifest.get("query_instruction", "")) == config.query_instruction
        and manifest_max_seq_length == config.max_seq_length
        and manifest_chunk_count == count_jsonl(chunks_path)
    )


def build_dense_index(
    chunks_path: str | Path,
    *,
    config: DenseIndexConfig | None = None,
    output_dir: str | Path | None = None,
    force: bool | None = None,
) -> dict[str, Path | int]:
    chunks_path = Path(chunks_path)
    dense_config = config or DenseIndexConfig()
    if not dense_config.enabled:
        log_block("Index Build", "Dense index skipped", "indexing.dense.enabled is false")
        return {"num_vectors": 0}

    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "Dense indexing requires faiss-cpu. Install project requirements or set "
            "indexing.dense.enabled: false for BM25-only retrieval."
        ) from exc

    should_force = dense_config.force if force is None else force
    target_dir = Path(output_dir) if output_dir else chunks_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = target_dir / dense_config.faiss_filename
    metadata_path = target_dir / dense_config.metadata_filename
    embeddings_path = target_dir / dense_config.embeddings_filename
    manifest_path = target_dir / dense_config.manifest_filename

    if (
        faiss_path.exists()
        and metadata_path.exists()
        and embeddings_path.exists()
        and cache_matches(manifest_path, chunks_path, dense_config)
        and count_jsonl(chunks_path) == count_jsonl(metadata_path)
        and not should_force
    ):
        log_block(
            "Index Build",
            "Dense index cache hit",
            f"FAISS index: {faiss_path}\nVectors: {count_jsonl(metadata_path)}",
        )
        return {
            "faiss_path": faiss_path,
            "metadata_path": metadata_path,
            "embeddings_path": embeddings_path,
            "manifest_path": manifest_path,
            "num_vectors": count_jsonl(metadata_path),
        }

    chunks = load_chunks(chunks_path)
    texts = [chunk["text"] for chunk in chunks]
    log_block(
        "Index Build",
        "Dense embedding",
        f"Chunks: {len(texts)}\nModel: {dense_config.model_name}\nDevice: {dense_config.device}",
    )
    embeddings = embed_passages(texts, dense_config)

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)

    faiss.write_index(index, str(faiss_path))
    np.save(embeddings_path, embeddings)
    write_metadata(metadata_path, chunks)
    write_manifest(manifest_path, chunks_path, dense_config, int(index.ntotal), int(embeddings.shape[1]))

    log_block(
        "Index Build",
        "Dense index completed",
        "\n".join(
            [
                f"FAISS index: {faiss_path}",
                f"Metadata: {metadata_path}",
                f"Embeddings: {embeddings_path}",
                f"Manifest: {manifest_path}",
                f"Vectors: {index.ntotal}",
            ]
        ),
    )
    return {
        "faiss_path": faiss_path,
        "metadata_path": metadata_path,
        "embeddings_path": embeddings_path,
        "manifest_path": manifest_path,
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
