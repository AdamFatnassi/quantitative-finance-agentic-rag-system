"""Split cleaned earnings-transcript turns into retrieval-sized chunks."""

from __future__ import annotations

import hashlib
import logging
import uuid

import pandas as pd

from src.rag.chunking.meta_data_schema import ChunkPayload, TranscriptMetadata
from src.rag.chunking.sec_chunker import ChunkerConfig, TextChunker, TokenCounter

logger = logging.getLogger(__name__)


def _context_header(row: pd.Series) -> str:
    ticker = str(row["ticker"])
    company = row.get("company_name")
    company_name = (
        str(company)
        if isinstance(company, str) and company.strip()
        else ticker
    )
    parts = [f"{company_name} ({ticker})", f"Earnings call {row['fiscal_quarter']}"]
    section = row.get("section")
    if isinstance(section, str) and section:
        parts.append(section.replace("_", " "))
    speaker_role = row.get("speaker_role")
    speaker_name = row.get("speaker_name")
    speaker = " - ".join(
        value for value in (speaker_role, speaker_name)
        if isinstance(value, str) and value
    )
    if speaker:
        parts.append(speaker)
    return " | ".join(parts)


class EarningsTranscriptChunker:
    """Chunk cleaned transcript turns and preserve validated source metadata."""

    def __init__(
        self,
        config: ChunkerConfig = ChunkerConfig(),
        token_counter: TokenCounter | None = None,
    ):
        self.cfg = config
        self.splitter = TextChunker(config, token_counter)
        self.in_path = (
            config.lake_root / "processed" / "earnings-transcripts" / "transcripts_clean.parquet"
        )
        self.out_path = (
            config.lake_root / "processed" / "earnings-transcripts" / "chunks.parquet"
        )

    def chunk_frame(self, turns: pd.DataFrame) -> pd.DataFrame:
        required = {
            "document_id", "turn_id", "ticker", "fiscal_quarter", "available_at",
            "text", "metadata",
        }
        missing = required - set(turns.columns)
        if missing:
            raise ValueError(f"input is missing columns: {sorted(missing)}")

        rows: list[dict] = []
        for _, turn in turns.iterrows():
            if not isinstance(turn["text"], str) or not turn["text"].strip():
                logger.warning("empty transcript turn, skipped: %s", turn["document_id"])
                continue

            metadata = TranscriptMetadata.model_validate(turn["metadata"])
            header = _context_header(turn)
            budget = max(
                self.cfg.max_tokens - self.splitter.count(header + "\n\n"),
                self.cfg.max_tokens // 2,
            )
            pieces = self.splitter.chunk(turn["text"], budget)
            for index, piece in enumerate(pieces):
                turn_id = str(turn["turn_id"])
                chunk_id = f"{turn_id}#{index:03d}"
                payload = ChunkPayload.model_validate({
                    "chunk_id": chunk_id,
                    "document_id": turn_id,
                    "text": piece,
                    "word_count": len(piece.split()),
                    "chunk_index": index,
                    "total_chunks": len(pieces),
                    "metadata": metadata,
                }).model_dump(mode="json")
                embedding_text = f"{header}\n\n{piece}"
                rows.append({
                    **payload,
                    "turn_id": turn_id,
                    "point_id": str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"earnings-transcript:{chunk_id}")
                    ),
                    "n_chunks": len(pieces),
                    "context_header": header,
                    "embedding_text": embedding_text,
                    "n_chars": len(piece),
                    "n_tokens_est": self.splitter.count(embedding_text),
                    "content_hash": hashlib.sha1(piece.encode("utf-8")).hexdigest(),
                    **metadata.model_dump(mode="json"),
                })
        return pd.DataFrame(rows)

    def run(self, write: bool = True) -> pd.DataFrame:
        if not self.in_path.exists():
            raise FileNotFoundError(
                f"{self.in_path} not found - run earnings_transcript_cleaner.py first."
            )
        turns = pd.read_parquet(self.in_path)
        chunks = self.chunk_frame(turns)
        logger.info("%d transcript turns -> %d chunks", len(turns), len(chunks))
        estimated_share = (
            float(chunks["availability_estimated"].astype(bool).mean()) if not chunks.empty else 0.0
        )
        logger.info("Transcript chunks with estimated availability: %.2f%%", estimated_share * 100)
        if write:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            chunks.to_parquet(self.out_path, index=False)
            logger.info("wrote %s", self.out_path)
        return chunks
