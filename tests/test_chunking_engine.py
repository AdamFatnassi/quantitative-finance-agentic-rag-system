"""Unit tests for structure-aware text chunking and SEC chunker orchestration."""

import pandas as pd
from datetime import date

from src.rag.chunking.sec_chunker import (
    ChunkerConfig,
    TextChunker,
    context_header,
    fine_split,
    is_table_like,
    point_id,
    SecFilingChunker,
)


def test_is_table_like_identifies_piped_rows():
    table_text = (
        "Item | Q1 2024 | Q1 2023\n"
        "Revenue | $90.8B | $94.8B\n"
        "Net Income | $23.6B | $24.1B"
    )
    assert is_table_like(table_text) is True

    prose_text = (
        "During the quarter ended March 31, 2024, our revenue was impacted "
        "by macroeconomic factors. However, demand remained resilient across key markets."
    )
    assert is_table_like(prose_text) is False


def test_fine_split_preserves_table_rows():
    table_text = "Revenue | $90.8B\nOperating Income | $27.9B\nNet Income | $23.6B"
    units = fine_split(table_text)
    assert len(units) == 3
    assert units[0] == ("Revenue | $90.8B", "\n")
    assert units[1] == ("Operating Income | $27.9B", "\n")


def test_fine_split_splits_prose_on_sentences():
    prose = "First sentence completed. Second sentence here! Third sentence follows?"
    units = fine_split(prose)
    assert len(units) == 3
    assert units[0][0] == "First sentence completed."
    assert units[1][0] == "Second sentence here!"
    assert units[2][0] == "Third sentence follows?"


def test_text_chunker_packs_and_respects_token_limit():
    cfg = ChunkerConfig(max_tokens=60, overlap_tokens=0, min_chunk_tokens=0, chars_per_token=4.0)
    chunker = TextChunker(cfg)
    
    paragraphs = "\n\n".join([
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau.",
        "Second long paragraph with plenty of additional words to ensure token boundary triggers.",
        "Third long paragraph providing further textual data to test multi-chunk splitting behavior.",
    ])
    chunks = chunker.chunk(paragraphs)
    
    assert len(chunks) > 1
    for chunk in chunks:
        # Every chunk should fit within max_tokens
        assert chunker.count(chunk) <= cfg.max_tokens


def test_text_chunker_trailing_merge():
    # When trailing chunk has fewer than min_chunk_tokens, it should merge into previous chunk
    cfg = ChunkerConfig(
        max_tokens=60,
        overlap_tokens=0,
        min_chunk_tokens=20,
        chars_per_token=4.0,
        merge_overflow=1.5,
    )
    chunker = TextChunker(cfg)
    text = (
        "This is a substantial first section of text that takes up most of the token budget "
        "and leaves just a very small amount of text remaining.\n\nShort tail."
    )
    chunks = chunker.chunk(text)
    # The tiny tail should be merged instead of forming an isolated 2nd chunk
    assert len(chunks) == 1
    assert "Short tail." in chunks[0]


def test_context_header_formatting():
    row_with_section = pd.Series({
        "company_name": "Apple Inc.",
        "ticker": "AAPL",
        "form": "10-Q",
        "filing_date": date(2024, 5, 3),
        "section_path": ["PART I", "Item 2"],
        "report_period_end": date(2024, 3, 30),
    })
    header = context_header(row_with_section)
    assert "Apple Inc. (AAPL)" in header
    assert "Form 10-Q filed 2024-05-03" in header
    assert "PART I > Item 2" in header
    assert "Period ended 2024-03-30" in header

    row_with_item = pd.Series({
        "company_name": "Microsoft Corp",
        "ticker": "MSFT",
        "form": "8-K",
        "filing_date": date(2024, 4, 25),
        "section_path": [],
        "item_code": "2.02",
        "item_label": "Results of Operations",
        "report_period_end": None,
    })
    header_8k = context_header(row_with_item)
    assert "Microsoft Corp (MSFT)" in header_8k
    assert "Item 2.02 - Results of Operations" in header_8k


def test_point_id_deterministic():
    id1 = point_id("AAPL:0001#000")
    id2 = point_id("AAPL:0001#000")
    id3 = point_id("AAPL:0001#001")
    assert id1 == id2
    assert id1 != id3
