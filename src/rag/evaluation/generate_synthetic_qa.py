"""Generate a RAG evaluation golden set from the project's chunk records.

Run from the repository root with:
    python -m src.rag.evaluation.generate_synthetic_qa

By default, chunks are read from data/lake/processed/8k-sec/chunks.parquet
and the golden set is written to data/golden_set/synthetic_qa.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS_PATH = (
    REPOSITORY_ROOT / "data" / "lake" / "processed" / "8k-sec" / "chunks.parquet"
)
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "data" / "golden_set" / "synthetic_qa.jsonl"

CHUNK_ONLY_FIELDS = {
    "chunk_id",
    "document_id",
    "text",
    "word_count",
    "chunk_index",
    "total_chunks",
    "n_chunks",
    "point_id",
    "context_header",
    "embedding_text",
    "n_chars",
    "n_tokens_est",
    "content_hash",
    "metadata",
}


class QAPair(BaseModel):
    question: str = Field(
        description=(
            "A realistic query a financial analyst or trading algorithm would ask, "
            "referencing specific companies or events."
        )
    )
    ground_truth: str = Field(
        description="The exact answer to the question, based strictly on the provided text."
    )


def _read_metadata(value: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    """Read nested chunk metadata, or recover it from flattened chunker columns."""
    if isinstance(value, str):
        value = json.loads(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("chunk metadata must be a mapping or a JSON object")

    metadata = dict(value)
    if not metadata:
        metadata = {
            key: item for key, item in record.items() if key not in CHUNK_ONLY_FIELDS
        }
    return metadata


def normalize_chunks(
    chunks: pd.DataFrame | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize project chunk rows and the original list-of-dicts format."""
    if isinstance(chunks, pd.DataFrame):
        records = chunks.to_dict(orient="records")
    else:
        records = list(chunks)

    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"chunk at index {index} must be a mapping")
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"chunk at index {index} must contain non-empty text")
        normalized.append(
            {
                "chunk_id": record.get("chunk_id"),
                "document_id": record.get("document_id"),
                "text": text,
                "metadata": _read_metadata(record.get("metadata"), record),
            }
        )
    return normalized


def generate_golden_dataset(
    chunks: pd.DataFrame | Iterable[Mapping[str, Any]],
    model_name: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """Create one RAG evaluation Q&A record per text chunk."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    normalized_chunks = normalize_chunks(chunks)
    llm = ChatOpenAI(model=model_name, temperature=0.2)
    structured_llm = llm.with_structured_output(QAPair)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert quantitative researcher building an evaluation dataset.
Given a text chunk and its metadata, generate ONE highly specific question that can
be answered definitively by the text.

Rules:
- Write a realistic user query about a financial fact, risk, or corporate event.
- Use the company name or ticker when available; avoid vague pronouns.
- Do not use facts that are absent from the chunk.
- The ground truth must answer the question using only the chunk.""",
            ),
            ("human", "Metadata: {metadata}\n\nChunk text: {text}"),
        ]
    )
    chain = prompt | structured_llm

    dataset = []
    failures = []
    logger.info("Generating synthetic Q&A pairs for %d chunks", len(normalized_chunks))
    for chunk in normalized_chunks:
        try:
            result = chain.invoke(
                {
                    "metadata": json.dumps(chunk["metadata"], default=str),
                    "text": chunk["text"],
                }
            )
            dataset.append(
                {
                    "question": result.question,
                    "ground_truth": result.ground_truth,
                    "contexts": [chunk["text"]],
                    "metadata": chunk["metadata"],
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                }
            )
        except Exception:
            logger.exception("Failed to generate Q&A for chunk %s", chunk["chunk_id"])
            failures.append(chunk["chunk_id"])

    if failures:
        logger.warning(
            "Q&A generation failed for %d of %d chunks",
            len(failures),
            len(normalized_chunks),
        )
    if not dataset and failures:
        raise RuntimeError("Q&A generation failed for every chunk")
    return pd.DataFrame(
        dataset,
        columns=[
            "question",
            "ground_truth",
            "contexts",
            "metadata",
            "chunk_id",
            "document_id",
        ],
    )


def _read_chunks(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported chunks file format: {path.suffix}")


def _write_dataset(dataset: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        dataset.to_parquet(path, index=False)
    elif suffix == ".csv":
        csv_dataset = dataset.copy()
        for column in ("contexts", "metadata"):
            csv_dataset[column] = csv_dataset[column].map(
                lambda value: json.dumps(value, default=str)
            )
        csv_dataset.to_csv(path, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        dataset.to_json(path, orient="records", lines=True, force_ascii=False)
    elif suffix == ".json":
        dataset.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError(f"unsupported output file format: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic RAG golden set from chunk records."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help=f"Chunk file (.parquet, .csv, .json, or .jsonl); default: {DEFAULT_CHUNKS_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Golden-set file; default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.input.is_file():
        raise FileNotFoundError(f"chunk file not found: {args.input}")
    chunks = _read_chunks(args.input)
    dataset = generate_golden_dataset(chunks, model_name=args.model)
    _write_dataset(dataset, args.output)
    logger.info("Wrote %d Q&A pairs to %s", len(dataset), args.output)


if __name__ == "__main__":
    main()
