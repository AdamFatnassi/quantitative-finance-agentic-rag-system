from datetime import date

import pandas as pd

from src.rag.chunking.sec_chunker import ChunkerConfig, SecFilingChunker
from src.rag.cleaning.sec_cleaner import TARGET_ITEMS, extract_report_sections, select_filings


def test_report_sections_keep_part_and_item_paths() -> None:
    body = (
        "PART I - FINANCIAL INFORMATION\n"
        "Item 1. Financial Statements\nRevenue increased.\n"
        "Item 2. Management's Discussion\nMargins improved.\n"
        "PART II - OTHER INFORMATION\n"
        "Item 1A. Risk Factors\nMarket risks changed.\n"
    )

    sections = extract_report_sections(body)

    assert [section["section_path"] for section in sections] == [
        "PART I FINANCIAL INFORMATION > Item 1",
        "PART I FINANCIAL INFORMATION > Item 2",
        "PART II OTHER INFORMATION > Item 1A",
    ]
    assert "Margins improved." in sections[1]["text"]


def test_selection_keeps_10k_and_10q_without_8k_items() -> None:
    manifest = pd.DataFrame(
        [
            {"ticker": "AAA", "form": "10-K", "items": "", "published_at": pd.Timestamp("2024-03-01", tz="UTC")},
            {"ticker": "BBB", "form": "10-Q", "items": "", "published_at": pd.Timestamp("2024-03-01", tz="UTC")},
            {"ticker": "CCC", "form": "8-K", "items": "2.02", "published_at": pd.Timestamp("2024-03-01", tz="UTC")},
            {"ticker": "DDD", "form": "8-K", "items": "", "published_at": pd.Timestamp("2024-03-01", tz="UTC")},
        ]
    )

    selected = select_filings(manifest, TARGET_ITEMS)

    assert selected["form"].tolist() == ["10-K", "10-Q", "8-K"]
    assert selected["items_matched"].tolist() == [[], [], ["2.02"]]


def test_selection_supports_explicit_backtest_derived_date_range() -> None:
    manifest = pd.DataFrame([
        {
            "ticker": "AAA",
            "form": "10-K",
            "items": "",
            "published_at": pd.Timestamp("2023-12-31T20:00:00Z"),
        },
        {
            "ticker": "AAA",
            "form": "10-K",
            "items": "",
            "published_at": pd.Timestamp("2024-01-01T20:00:00Z"),
        },
        {
            "ticker": "AAA",
            "form": "10-K",
            "items": "",
            "published_at": pd.Timestamp("2024-02-01T20:00:00Z"),
        },
    ])

    selected = select_filings(
        manifest,
        TARGET_ITEMS,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )

    assert selected["published_at"].tolist() == [
        pd.Timestamp("2024-01-01T20:00:00Z"),
    ]


def test_chunk_payload_schema_and_parquet_round_trip(tmp_path) -> None:
    docs = pd.DataFrame(
        [
            {
                "doc_id": "AAA:0001:part-i-item-2",
                "source_type": "sec_10q",
                "ticker": "AAA",
                "cik": 12345,
                "company_name": "Example Corp",
                "accession_number": "0000012345-24-000001",
                "form": "10-Q",
                "filing_date": date(2024, 5, 1),
                "report_period_end": date(2024, 3, 31),
                "fiscal_year": 2024,
                "fiscal_period": "Q1",
                "published_at": "2024-05-01T20:00:00+00:00",
                "available_at": "2024-05-01T20:00:00+00:00",
                "availability_estimated": False,
                "section_type": "section",
                "items_all": [],
                "items_matched": [],
                "part_type": "section",
                "item_code": "2",
                "item_label": "Management's Discussion",
                "section_path": ["PART I", "Item 2"],
                "exhibit_type": None,
                "parse_mode": "sections",
                "source_url": "https://www.sec.gov/Archives/example",
                "raw_path": "raw/example.txt",
                "text": "Revenue increased during the quarter.",
            }
        ]
    )
    chunker = SecFilingChunker(
        ChunkerConfig(max_tokens=120, overlap_tokens=10, min_chunk_tokens=0)
    )

    chunks = chunker.chunk_frame(docs)

    assert len(chunks) == 1
    chunk = chunks.iloc[0]
    assert chunk["document_id"] == "AAA:0001:part-i-item-2"
    assert chunk["word_count"] == len(chunk["text"].split())
    assert chunk["metadata"]["source_type"] == "sec_10q"
    assert chunk["metadata"]["available_at"] == "2024-05-01T20:00:00Z"
    assert chunk["metadata"]["availability_estimated"] is False
    assert chunk["metadata"]["section_type"] == "section"
    assert chunk["metadata"]["form_type"] == "10-Q"
    assert chunk["metadata"]["section_path"] == ["PART I", "Item 2"]
    assert chunk["section_path"] == ["PART I", "Item 2"]

    output = tmp_path / "chunks.parquet"
    chunks.to_parquet(output, index=False)
    loaded = pd.read_parquet(output)
    assert loaded.iloc[0]["metadata"]["accession_number"] == "0000012345-24-000001"
