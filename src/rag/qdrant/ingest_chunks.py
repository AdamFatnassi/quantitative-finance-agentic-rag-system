"""Idempotently embed and index cleaned SEC and transcript chunks in Qdrant."""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
from qdrant_client.http import models as qm

from src.rag.qdrant.qdrant_lifecycle import get_client
from src.rag.scraping.universe import load_universe
from src.rag.coverage import build_ingestion_status

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
CHUNK_FILES = (
    Path("processed/8k-sec/chunks.parquet"),
    Path("processed/earnings-transcripts/chunks.parquet"),
)
PAYLOAD_FIELDS = (
    "ticker",       # kept for backward compat; use tickers for filtering
    "tickers",      # list of all ticker symbols that share this filing (e.g. ["GOOG","GOOGL"])
    "cik",
    "available_at",
    "published_at",
    "section_type",
    "form_type",
    "item_code",
    "accession_number",
    "transcript_id",
    "availability_estimated",
    "source_type",
    "chunk_id",
    "document_id",
    "chunk_index",
    "text",
)


@dataclass(frozen=True)
class IndexerConfig:
    lake_root: Path = Path("data/lake")
    collection_name: str = field(
        default_factory=lambda: os.getenv("RAG_QDRANT_COLLECTION", "financial_text_chunks")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
    )
    sparse_model: str = field(
        default_factory=lambda: os.getenv("RAG_SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
    )
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333"))
    )
    batch_size: int = 64

    def __post_init__(self) -> None:
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not self.embedding_model.strip() or not self.sparse_model.strip():
            raise ValueError("dense and sparse embedding model names must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


class Encoder(Protocol):
    def embed(self, texts: list[str]) -> tuple[list, list]:
        """Return aligned dense vectors and sparse embeddings."""


class FastEmbedEncoder:
    def __init__(self, dense_model: str, sparse_model: str):
        try:
            from fastembed import SparseTextEmbedding, TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Install qdrant-client[fastembed] to run chunk embedding."
            ) from exc
        self._dense = TextEmbedding(model_name=dense_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)

    def embed(self, texts: list[str]) -> tuple[list, list]:
        dense = list(self._dense.embed(texts))
        sparse = list(self._sparse.embed(texts))
        if len(dense) != len(texts) or len(sparse) != len(texts):
            raise ValueError("embedding models returned a mismatched number of vectors")
        return dense, sparse


def _as_utc_iso(value: object, *, field_name: str, required: bool = False) -> str | None:
    if value is None or pd.isna(value):
        if required:
            raise ValueError(f"chunk has no {field_name}")
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"chunk {field_name} must be timezone-aware")
    return timestamp.tz_convert("UTC").isoformat()


def _nonempty_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _payload_value(value: object) -> object:
    if value is None:
        return None
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def deduplicate_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate rows that share the same (cik, accession_number, document_id,
    chunk_index) natural key — e.g. GOOG and GOOGL indexing the same Alphabet filing.

    The first row is kept; all ticker symbols from the duplicate group are merged
    into a sorted list stored in a new ``tickers`` column.  The scalar ``ticker``
    column retains the value from the first row (alphabetical order within each
    group) for backward compatibility.

    For non-SEC chunks (transcript rows) that have no ``accession_number``, the
    key is (transcript_id, document_id, chunk_index); these are passed through
    unchanged but still get a ``tickers`` column.
    """
    if df.empty:
        df = df.copy()
        if "tickers" not in df.columns:
            df["tickers"] = pd.Series(dtype=object)
        return df

    df = df.copy()
    # Normalise the ticker column so the sort is stable
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Build the deduplication key
    acc_col = df.get("accession_number", pd.Series("", index=df.index)).fillna("")
    tx_col  = df.get("transcript_id",   pd.Series("", index=df.index)).fillna("")
    doc_col = (
        df.get("document_id", pd.Series("", index=df.index))
        .fillna(df.get("doc_id", pd.Series("", index=df.index)).fillna(""))
    )
    cik_col = df.get("cik", pd.Series(0, index=df.index)).fillna(0).astype(str)

    df["_dedup_key"] = (
        cik_col + "|" + acc_col + "|" + tx_col + "|" + doc_col + "|"
        + df["chunk_index"].astype(str)
    )

    # Sort so that within each duplicate group tickers are in alphabetical order
    df = df.sort_values(["_dedup_key", "ticker"]).reset_index(drop=True)

    # Aggregate: collect tickers list per group
    tickers_map: dict[str, list[str]] = (
        df.groupby("_dedup_key", sort=False)["ticker"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )
    df["tickers"] = df["_dedup_key"].map(tickers_map)

    # Keep only the first occurrence of each key
    df = df.drop_duplicates(subset=["_dedup_key"], keep="first")
    df = df.drop(columns=["_dedup_key"]).reset_index(drop=True)

    n_removed = len(df) - df.shape[0]  # always 0 here; logging uses pre-drop length
    return df


def _point_id(row: dict) -> str:
    document_id = _nonempty_string(row.get("document_id")) or _nonempty_string(
        row.get("doc_id")
    )
    if document_id is None:
        raise ValueError("each chunk must include document_id or doc_id")
    try:
        chunk_index = int(row["chunk_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"chunk for {document_id!r} has no valid chunk_index") from exc
    stable_key = f"{document_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"financial-text:{stable_key}"))


def _payload(row: dict, point_id: str) -> dict:
    payload = {key: _payload_value(row.get(key)) for key in PAYLOAD_FIELDS
               if key != "tickers"}   # tickers is a list; handled separately below
    payload["chunk_id"] = _nonempty_string(row.get("chunk_id")) or point_id
    payload["document_id"] = (
        _nonempty_string(row.get("document_id"))
        or _nonempty_string(row.get("doc_id"))
    )
    payload["chunk_index"] = int(row["chunk_index"])
    payload["ticker"] = str(row["ticker"]).strip().upper()
    # tickers: list of all ticker symbols that share this filing
    raw_tickers = row.get("tickers")
    if isinstance(raw_tickers, list) and raw_tickers:
        payload["tickers"] = sorted({str(t).strip().upper() for t in raw_tickers})
    else:
        payload["tickers"] = [payload["ticker"]]
    cik = row.get("cik")
    payload["cik"] = int(cik) if cik is not None and not pd.isna(cik) else None
    payload["available_at"] = _as_utc_iso(
        row.get("available_at"), field_name="available_at", required=True
    )
    payload["published_at"] = _as_utc_iso(row.get("published_at"), field_name="published_at")
    estimated = row.get("availability_estimated")
    if estimated is None or pd.isna(estimated):
        raise ValueError(f"chunk {payload['chunk_id']} has no availability_estimated flag")
    payload["availability_estimated"] = bool(estimated)
    if not isinstance(payload["text"], str) or not payload["text"].strip():
        raise ValueError(f"chunk {payload['chunk_id']} has no text")
    return payload


def _ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    embedding_model: str,
    sparse_model: str,
) -> None:
    model_metadata = {
        "dense_embedding_model": embedding_model,
        "sparse_embedding_model": sparse_model,
    }
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR: qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR: qm.SparseVectorParams(index=qm.SparseIndexParams())
            },
            metadata=model_metadata,
        )
    else:
        collection = client.get_collection(collection_name)
        metadata = collection.config.metadata or {}
        if any(metadata.get(key) != value for key, value in model_metadata.items()):
            raise ValueError(
                f"Qdrant collection {collection_name!r} was built with different "
                "embedding models; use a new collection before switching models"
            )
        config = collection.config.params
        vectors = config.vectors
        dense_config = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
        if dense_config is None or dense_config.size != vector_size:
            raise ValueError(
                f"Qdrant collection {collection_name!r} is incompatible with "
                f"embedding dimension {vector_size}"
            )
        if not config.sparse_vectors or SPARSE_VECTOR not in config.sparse_vectors:
            raise ValueError(
                f"Qdrant collection {collection_name!r} has no {SPARSE_VECTOR!r} vector"
            )

    schema = client.get_collection(collection_name).payload_schema or {}
    for key, field_schema in (
        ("ticker",      qm.PayloadSchemaType.KEYWORD),
        ("tickers",     qm.PayloadSchemaType.KEYWORD),   # array field — Qdrant indexes each element
        ("available_at", qm.PayloadSchemaType.DATETIME),
        ("section_type", qm.PayloadSchemaType.KEYWORD),
    ):
        if key not in schema:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=key,
                field_schema=field_schema,
            )


def validate_collection_models(
    client: QdrantClient,
    collection_name: str,
    embedding_model: str,
    sparse_model: str,
) -> None:
    """Reject dense/sparse model changes that would mix incompatible vectors."""
    collection = client.get_collection(collection_name)
    metadata = collection.config.metadata or {}
    if (
        metadata.get("dense_embedding_model") != embedding_model
        or metadata.get("sparse_embedding_model") != sparse_model
    ):
        raise ValueError(
            f"Qdrant collection {collection_name!r} was built with different "
            "embedding models; use the collection's configured models or a new collection"
        )


def _existing_point_ids(
    client: QdrantClient, collection_name: str, ids: list[str], batch_size: int
) -> set[str]:
    existing: set[str] = set()
    for start in range(0, len(ids), batch_size):
        points = client.retrieve(
            collection_name=collection_name,
            ids=ids[start : start + batch_size],
            with_payload=False,
            with_vectors=False,
        )
        existing.update(str(point.id) for point in points)
    return existing


def ingest_frame(
    chunks: pd.DataFrame,
    client: QdrantClient,
    config: IndexerConfig = IndexerConfig(),
    encoder: Encoder | None = None,
) -> dict[str, int | str]:
    """Index new chunk rows only; deterministic point IDs make reruns safe."""
    required = {
        "ticker", "available_at", "availability_estimated", "chunk_index",
        "document_id", "text",
    }
    missing = required - set(chunks.columns)
    if missing:
        raise ValueError(f"chunk data is missing columns: {sorted(missing)}")
    if chunks.empty:
        count = (
            client.count(collection_name=config.collection_name, exact=True).count
            if client.collection_exists(config.collection_name)
            else 0
        )
        return {"indexed": 0, "skipped": 0, "point_count": count}
    if client.collection_exists(config.collection_name):
        validate_collection_models(
            client,
            config.collection_name,
            config.embedding_model,
            config.sparse_model,
        )

    rows = chunks.to_dict(orient="records")
    prepared = [(_point_id(row), _payload(row, _point_id(row)), row) for row in rows]
    if len({point_id for point_id, _, _ in prepared}) != len(prepared):
        raise ValueError("chunk input contains duplicate stable point IDs")

    if encoder is None:
        encoder = FastEmbedEncoder(config.embedding_model, config.sparse_model)

    skipped = 0
    indexed = 0
    for start in range(0, len(prepared), config.batch_size):
        batch = prepared[start : start + config.batch_size]
        ids = [point_id for point_id, _, _ in batch]
        existing = (
            _existing_point_ids(client, config.collection_name, ids, config.batch_size)
            if client.collection_exists(config.collection_name)
            else set()
        )
        new_batch = [record for record in batch if record[0] not in existing]
        skipped += len(batch) - len(new_batch)
        if not new_batch:
            continue

        embedding_texts = [
            _nonempty_string(record[2].get("embedding_text")) or record[1]["text"]
            for record in new_batch
        ]
        dense_vectors, sparse_vectors = encoder.embed(embedding_texts)
        if len(dense_vectors) != len(new_batch) or len(sparse_vectors) != len(new_batch):
            raise ValueError("embedding count does not match new chunk count")
        vector_size = len(dense_vectors[0])
        if vector_size < 1:
            raise ValueError("dense embedding vectors must not be empty")
        _ensure_collection(
            client,
            config.collection_name,
            vector_size,
            config.embedding_model,
            config.sparse_model,
        )

        points = []
        for (point_id, payload, _), dense, sparse in zip(
            new_batch, dense_vectors, sparse_vectors
        ):
            indices = [int(index) for index in sparse.indices]
            values = [float(value) for value in sparse.values]
            points.append(qm.PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR: [float(value) for value in dense],
                    SPARSE_VECTOR: qm.SparseVector(indices=indices, values=values),
                },
                payload=payload,
            ))
        client.upsert(collection_name=config.collection_name, points=points, wait=True)
        indexed += len(points)

    point_count = client.count(collection_name=config.collection_name, exact=True).count
    return {
        "indexed": indexed,
        "skipped": skipped,
        "point_count": point_count,
        "embedding_model": config.embedding_model,
        "sparse_model": config.sparse_model,
    }


def load_chunk_frames(lake_root: Path, tickers: list[str] | None = None) -> pd.DataFrame:
    wanted = (
        set(load_universe(lake_root)["ticker"])
        if tickers is None
        else {ticker.strip().upper() for ticker in tickers}
    )
    frames = []
    for relative_path in CHUNK_FILES:
        path = lake_root / relative_path
        if path.is_file():
            frame = pd.read_parquet(path)
            frame = frame[frame["ticker"].astype(str).str.upper().isin(wanted)]
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No cleaned chunk Parquet files with matching records were found under "
            f"{lake_root / 'processed'}"
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def persist_ingestion_status(engine, status: pd.DataFrame) -> int:
    """Upsert per-ticker/source availability bounds into the relational status table."""
    if status.empty:
        return 0
    from sqlalchemy import text

    statement = text(
        "INSERT INTO ingestion_status "
        "(ticker, source, earliest_available_at, latest_available_at, chunk_count, status, status_updated_at) "
        "VALUES (:ticker, :source, :earliest_available_at, :latest_available_at, "
        ":chunk_count, :status, now()) "
        "ON CONFLICT (ticker, source) DO UPDATE SET "
        "earliest_available_at = EXCLUDED.earliest_available_at, "
        "latest_available_at = EXCLUDED.latest_available_at, "
        "chunk_count = EXCLUDED.chunk_count, "
        "status = EXCLUDED.status, "
        "status_updated_at = now()"
    )
    rows = [
        {
            key: None if value is pd.NaT or pd.isna(value) else value
            for key, value in row.items()
        }
        for row in status.to_dict(orient="records")
    ]
    with engine.begin() as connection:
        connection.execute(statement, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    parser.add_argument("--tickers", nargs="+", help="Optional test-only ticker filter")
    parser.add_argument("--collection", default=os.getenv(
        "RAG_QDRANT_COLLECTION", "financial_text_chunks"
    ))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="database URL for ingestion_status (default: DATABASE_URL)",
    )
    parser.add_argument(
        "--status-parquet-only",
        action="store_true",
        help="skip the database table and write only processed/ingestion_status.parquet",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the post-ingestion validation stage (not recommended for production)",
    )
    args = parser.parse_args()
    if not args.database_url and not args.status_parquet_only:
        parser.error(
            "set DATABASE_URL for ingestion_status or explicitly pass --status-parquet-only"
        )

    config = IndexerConfig(
        lake_root=args.lake_root,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )

    chunks = load_chunk_frames(config.lake_root, args.tickers)
    n_before = len(chunks)
    chunks = deduplicate_chunks(chunks)
    n_removed = n_before - len(chunks)
    if n_removed:
        logger.info(
            "Deduplication removed %d duplicate chunk row(s) (shared CIK filings, e.g. GOOG/GOOGL).",
            n_removed,
        )
    client = get_client(config.qdrant_host, config.qdrant_port)
    try:
        result = ingest_frame(chunks, client, config)
        logger.info("Qdrant chunk indexing complete: %s", result)
    finally:
        client.close()
    status_tickers = args.tickers or load_universe(config.lake_root)["ticker"].tolist()
    status = build_ingestion_status(chunks, status_tickers)
    status_path = config.lake_root / "processed" / "ingestion_status.parquet"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status.to_parquet(status_path, index=False)
    if args.database_url:
        from sqlalchemy import create_engine

        engine = create_engine(args.database_url)
        try:
            persisted = persist_ingestion_status(engine, status)
        finally:
            engine.dispose()
        logger.info("Upserted %d ingestion_status row(s)", persisted)
    else:
        logger.warning("ingestion_status persisted to Parquet only: %s", status_path)

    # ---- Post-ingestion validation ----------------------------------------
    if args.no_validate:
        logger.warning("Post-ingestion validation skipped (--no-validate).")
        return

    from src.rag.validation import ValidationConfig, ValidationError, print_report, run_validation

    val_config = ValidationConfig(
        lake_root=config.lake_root,
        collection_name=config.collection_name,
        qdrant_host=config.qdrant_host,
        qdrant_port=config.qdrant_port,
        database_url=args.database_url or None,
        fail_on_violation=True,
    )
    try:
        report = run_validation(val_config)
        print_report(report)
    except ValidationError as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
