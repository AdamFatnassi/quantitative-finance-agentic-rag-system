"""
sec_chunker.py - split cleaned SEC filing sections into retrieval-sized chunks.

Responsibility: CHUNKING ONLY. Reads the cleaner's output, writes chunk rows.

Input : <lake_root>/processed/8k-sec/filings_clean.parquet   (from sec_cleaner.py)
Output: <lake_root>/processed/8k-sec/chunks.parquet          (one row per chunk)

Chunking strategy (structure-aware, packed greedily)
----------------------------------------------------
1. Atomic units, finest-needed granularity:
     paragraph  ->  (if too big)  table rows or sentences  ->  (if still too big) words
   Tables from press releases (rows joined by " | ") are split on row boundaries,
   never mid-row.
2. Units are packed into chunks of at most `max_tokens`, never splitting a unit that
   already fits. Chunks therefore end on paragraph / row / sentence boundaries.
3. Overlap: when a chunk closes, its last `overlap_tokens` worth of text (whole
   units, or the trailing sentences / rows of the last one) opens the next chunk.
4. A tiny trailing chunk (< `min_chunk_tokens` of new text) is merged into the
   previous chunk instead of being emitted alone.
5. Each chunk gets a context header (company, ticker, form, filing date, item) so
   the embedding of a short chunk still carries "who / what / when". The header is
   budgeted: header + chunk text stays within `max_tokens`.

Token counting: by default a fast estimate (characters / chars_per_token). To match your embedding model exactly, pass its tokenizer:
SecFilingChunker(cfg, token_counter=fn) with fn(text) -> int. The estimate is
deliberately conservative (rounds up).

Columns written
---------------
  chunk_id        "<doc_id>#<index>"          human-readable, unique
  point_id        UUID derived from chunk_id  stable: re-running never duplicates points
  document_id, word_count, chunk_index, total_chunks, n_chunks
  metadata         schema-validated nested filters/citations
  text            the chunk (use for display / citation)
  context_header  one-line header
  embedding_text  context_header + text       (what you should embed)
  n_chars, n_tokens_est, content_hash
  + SEC form, ticker/CIK, dates, item and section paths, source and provenance

Usage
-----
    python -m src.rag.chunking.sec_chunker --max-tokens 350 --overlap-tokens 50
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src.rag.chunking.meta_data_schema import ChunkPayload

logger = logging.getLogger(__name__)

RAW_DIRNAME = "8k-sec"                       # must match scraper / cleaner

TokenCounter = Callable[[str], int]
Unit = tuple[str, str]                       # (text, separator that precedes it)

METADATA_COLS = [
    "doc_id", "source_type", "ticker", "cik", "company_name", "accession_number",
    "form", "filing_date", "report_period_end", "fiscal_year", "fiscal_period",
    "published_at", "available_at", "availability_estimated", "section_type",
    "items_all", "items_matched", "part_type", "item_code",
    "item_label", "section_path", "exhibit_type", "parse_mode", "source_url", "raw_path",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkerConfig:
    lake_root: Path = Path("data/lake")
    max_tokens: int = 350            # header + chunk text; safe for 512-token embedding models
    overlap_tokens: int = 50
    min_chunk_tokens: int = 60       # trailing chunks with less new text are merged back
    chars_per_token: float = 4.0     # only used by the default estimator
    merge_overflow: float = 1.25     # a merged chunk may exceed max_tokens by at most this factor

    def __post_init__(self) -> None:
        if self.max_tokens < 50:
            raise ValueError("max_tokens must be >= 50")
        if not 0 <= self.overlap_tokens < self.max_tokens // 2:
            raise ValueError("overlap_tokens must be >= 0 and < max_tokens / 2")
        if self.min_chunk_tokens < 0 or self.chars_per_token <= 0 or self.merge_overflow < 1:
            raise ValueError("invalid min_chunk_tokens / chars_per_token / merge_overflow")


# ---------------------------------------------------------------------------
# Text splitting helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9("$])')
_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")


def is_table_like(text: str) -> bool:
    """True for blocks that are mostly table rows (the cleaner joins cells with ' | ')."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return len(lines) >= 2 and sum(" | " in ln for ln in lines) >= 0.6 * len(lines)


def fine_split(text: str) -> list[Unit]:
    """Split one oversized block into rows (tables) or sentences (prose)."""
    if is_table_like(text):
        parts, sep = text.split("\n"), "\n"
    else:
        parts, sep = _SENTENCE_SPLIT.split(text), " "
    return [(p.strip(), sep) for p in parts if p.strip()]


def join_units(units: list[Unit]) -> str:
    return "".join((sep if i else "") + text for i, (text, sep) in enumerate(units))


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

class TextChunker:
    def __init__(self, config: ChunkerConfig, token_counter: Optional[TokenCounter] = None):
        self.cfg = config
        cpt = config.chars_per_token
        self.count: TokenCounter = token_counter or (lambda s: max(1, math.ceil(len(s) / cpt)))

    # ----- public ----------------------------------------------------------

    def chunk(self, text: str, max_tokens: Optional[int] = None) -> list[str]:
        """Split `text` into chunks of at most `max_tokens` (default: config value)."""
        limit = max_tokens or self.cfg.max_tokens
        text = text.strip()
        if not text:
            return []
        if self.count(text) <= limit:
            return [text]

        units = self._to_units(text, limit)
        return [join_units(c) for c in self._pack(units, limit)]

    # ----- step 1: text -> units ------------------------------------------

    def _to_units(self, text: str, limit: int) -> list[Unit]:
        units: list[Unit] = []
        for paragraph in _PARAGRAPH_SPLIT.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if self.count(paragraph) <= limit:
                units.append((paragraph, "\n\n"))
                continue
            first = True
            for piece, sep in fine_split(paragraph):
                for part in self._fit(piece, limit):
                    units.append((part, "\n\n" if first else sep))
                    first = False
        return units

    def _fit(self, piece: str, limit: int) -> list[str]:
        """Guarantee every unit fits: split an oversized sentence / row on words, then characters."""
        if self.count(piece) <= limit:
            return [piece]
        parts: list[str] = []
        current = ""
        max_chars = max(1, int(limit * self.cfg.chars_per_token))
        for word in piece.split():
            while self.count(word) > limit:                       # one unbroken "word" too long
                if current:
                    parts.append(current)
                    current = ""
                parts.append(word[:max_chars])
                word = word[max_chars:]
            candidate = f"{current} {word}".strip()
            if current and self.count(candidate) > limit:
                parts.append(current)
                current = word
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts

    # ----- step 2: pack units into chunks ----------------------------------

    def _cost(self, unit: Unit) -> int:
        text, sep = unit
        return self.count(sep + text)

    def _pack(self, units: list[Unit], limit: int) -> list[list[Unit]]:
        chunks: list[list[Unit]] = []
        current: list[Unit] = []        # units in the open chunk (overlap + new)
        new: list[Unit] = []            # units in the open chunk that are NOT overlap
        used = 0

        for unit in units:
            cost = self._cost(unit)
            if current and used + cost > limit:
                chunks.append(current)
                current = self._overlap_tail(current)
                used = sum(self._cost(u) for u in current)
                if used + cost > limit:                           # overlap would crowd out the new unit
                    current, used = [], 0
                new = []
            current.append(unit)
            new.append(unit)
            used += cost

        if current:
            last_new_tokens = sum(self._cost(u) for u in new)
            if chunks and last_new_tokens < self.cfg.min_chunk_tokens:
                merged = chunks[-1] + new                          # overlap is already inside chunks[-1]
                if sum(self._cost(u) for u in merged) <= limit * self.cfg.merge_overflow:
                    chunks[-1] = merged
                    return chunks
            chunks.append(current)
        return chunks

    def _overlap_tail(self, closed: list[Unit]) -> list[Unit]:
        """Trailing text of a closed chunk (whole units, then trailing sentences / rows)."""
        budget = self.cfg.overlap_tokens
        if budget <= 0:
            return []
        tail: list[Unit] = []
        used = 0
        for unit in reversed(closed):
            cost = self._cost(unit)
            if used + cost <= budget:
                tail.append(unit)
                used += cost
                continue
            for piece in reversed(fine_split(unit[0])):           # unit too big: take its tail only
                piece_cost = self._cost(piece)
                if used + piece_cost > budget:
                    break
                tail.append(piece)
                used += piece_cost
            break
        tail.reverse()
        return [] if join_units(tail) == join_units(closed) else tail   # never repeat a whole chunk


# ---------------------------------------------------------------------------
# Document-level orchestration
# ---------------------------------------------------------------------------

def context_header(row: pd.Series) -> str:
    section_path = row.get("section_path")
    if isinstance(section_path, (list, tuple)) and section_path:
        where = " > ".join(str(part) for part in section_path)
    else:
        item_code = row.get("item_code")
        item_label = row.get("item_label")
        if pd.api.types.is_scalar(item_code) and not pd.isna(item_code):
            where = f"Item {item_code}"
            if pd.api.types.is_scalar(item_label) and not pd.isna(item_label):
                where += f" - {item_label}"
        elif pd.api.types.is_scalar(item_label) and not pd.isna(item_label):
            where = str(item_label)
        else:
            where = "Filing"

    company = row.get("company_name")
    ticker = row.get("ticker")
    if not pd.api.types.is_scalar(company) or pd.isna(company):
        company = ticker if pd.api.types.is_scalar(ticker) and not pd.isna(ticker) else "Unknown company"
    if not pd.api.types.is_scalar(ticker) or pd.isna(ticker):
        ticker = "unknown ticker"
    header = f"{company} ({ticker}) | Form {row['form']} filed {row['filing_date']} | {where}"
    report_period_end = row.get("report_period_end")
    if pd.api.types.is_scalar(report_period_end) and not pd.isna(report_period_end):
        header += f" | Period ended {report_period_end}"
    return header


def point_id(chunk_id: str) -> str:
    """Deterministic UUID: the same chunk always maps to the same vector-store point."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sec-8k:{chunk_id}"))


class SecFilingChunker:
    def __init__(self, config: ChunkerConfig = ChunkerConfig(), token_counter: Optional[TokenCounter] = None):
        self.cfg = config
        self.splitter = TextChunker(config, token_counter)
        self.in_path = config.lake_root / "processed" / RAW_DIRNAME / "filings_clean.parquet"
        self.out_path = config.lake_root / "processed" / RAW_DIRNAME / "chunks.parquet"

    def chunk_frame(self, docs: pd.DataFrame) -> pd.DataFrame:
        """Chunk every row of a cleaner-output DataFrame."""
        required = {
            "doc_id", "ticker", "cik", "company_name", "accession_number",
            "form", "filing_date", "published_at", "text",
        }
        missing = required - set(docs.columns)
        if missing:
            raise ValueError(f"input is missing columns: {sorted(missing)}")

        meta_cols = [c for c in METADATA_COLS if c in docs.columns]
        rows: list[dict] = []
        for _, doc in docs.iterrows():
            if not isinstance(doc["text"], str) or not doc["text"].strip():
                logger.warning("empty text, skipped: %s", doc["doc_id"])
                continue
            header = context_header(doc)
            # reserve room for the header (+ blank line) so header + chunk fits in max_tokens
            budget = max(self.cfg.max_tokens - self.splitter.count(header + "\n\n"),
                         self.cfg.max_tokens // 2)
            pieces = self.splitter.chunk(doc["text"], budget)
            form = str(doc["form"])
            source_type = doc.get("source_type")
            if not isinstance(source_type, str) or not source_type:
                source_type = (
                    "sec_8k" if form.startswith("8-K")
                    else "sec_10k" if form.startswith("10-K")
                    else "sec_10q" if form.startswith("10-Q")
                    else None
                )
            if source_type is None:
                raise ValueError(f"unsupported SEC form in chunk input: {form}")

            metadata = {}
            for column in meta_cols:
                if column == "doc_id":
                    continue
                value = doc[column]
                if pd.api.types.is_scalar(value) and pd.isna(value):
                    value = None
                elif column in {"items_all", "items_matched", "section_path"}:
                    value = list(value) if value is not None else []
                metadata[column] = value
            section_path = metadata.get("section_path")
            items_all = metadata.get("items_all")
            items_matched = metadata.get("items_matched")
            metadata.update({
                "source_type": source_type,
                "ticker": str(doc["ticker"]),
                "company_name": str(doc["company_name"]),
                "published_at": doc["published_at"],
                "available_at": doc["available_at"],
                "availability_estimated": bool(doc.get("availability_estimated", False)),
                "section_type": metadata.get("section_type") or metadata.get("part_type"),
                "form_type": form,
                "source_url": metadata.get("source_url"),
                "form": form,
                "filing_date": doc["filing_date"],
                "cik": int(doc["cik"]),
                "accession_number": str(doc["accession_number"]),
                "item_title": metadata.get("item_label"),
                "section_path": section_path if section_path is not None else [],
                "items_all": items_all if items_all is not None else [],
                "items_matched": items_matched if items_matched is not None else [],
            })
            for i, piece in enumerate(pieces):
                embedding_text = f"{header}\n\n{piece}"
                chunk_id = f"{doc['doc_id']}#{i:03d}"
                payload = ChunkPayload.model_validate({
                    "chunk_id": chunk_id,
                    "document_id": str(doc["doc_id"]),
                    "text": piece,
                    "word_count": len(piece.split()),
                    "chunk_index": i,
                    "total_chunks": len(pieces),
                    "metadata": metadata,
                }).model_dump(mode="json")
                rows.append({
                    **payload,
                    "chunk_id": chunk_id,
                    "document_id": str(doc["doc_id"]),
                    "word_count": len(piece.split()),
                    "point_id": point_id(chunk_id),
                    "chunk_index": i,
                    "total_chunks": len(pieces),
                    "n_chunks": len(pieces),
                    "text": piece,
                    "context_header": header,
                    "embedding_text": embedding_text,
                    "n_chars": len(piece),
                    "n_tokens_est": self.splitter.count(embedding_text),
                    "content_hash": hashlib.sha1(piece.encode("utf-8")).hexdigest(),
                    **metadata,
                    **{c: doc[c] for c in meta_cols},
                })
        return pd.DataFrame(rows)

    def run(self, write: bool = True) -> pd.DataFrame:
        if not self.in_path.exists():
            raise FileNotFoundError(f"{self.in_path} not found - run sec_cleaner.py first.")
        docs = pd.read_parquet(self.in_path)
        chunks = self.chunk_frame(docs)
        logger.info("%d documents -> %d chunks", len(docs), len(chunks))
        estimated_share = (
            float(chunks["availability_estimated"].astype(bool).mean()) if not chunks.empty else 0.0
        )
        logger.info("SEC chunks with estimated availability: %.2f%%", estimated_share * 100)
        if write:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            chunks.to_parquet(self.out_path, index=False)
            logger.info("wrote %s", self.out_path)
        return chunks


Sec8KChunker = SecFilingChunker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk cleaned SEC documents for embedding.")
    parser.add_argument("--lake-root", default="data/lake")
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--min-tokens", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ChunkerConfig(lake_root=Path(args.lake_root), max_tokens=args.max_tokens,
                        overlap_tokens=args.overlap_tokens, min_chunk_tokens=args.min_tokens)
    chunks = SecFilingChunker(cfg).run()

    if chunks.empty:
        print("No chunks produced (cleaner output is empty).")
        return
    print(f"\n{len(chunks)} chunks from {chunks['doc_id'].nunique()} documents")
    print(chunks["n_tokens_est"].describe().round(1).to_string())
    summary = chunks.assign(
        part=chunks["section_path"].map(
            lambda path: " > ".join(path) if isinstance(path, (list, tuple)) else str(path)
        )
    )
    by_part = summary.groupby(["ticker", "form", "part"], dropna=False) \
                    .agg(chunks=("chunk_id", "size"), tokens=("n_tokens_est", "sum"))
    print(by_part.to_string())


if __name__ == "__main__":
    main()
