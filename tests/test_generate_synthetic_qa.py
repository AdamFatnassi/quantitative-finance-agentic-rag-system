"""Tests for adapting project chunk records into golden-set records."""

import json

import pandas as pd
import pytest

from src.rag.evaluation.generate_synthetic_qa import (
    _write_dataset,
    normalize_chunks,
)


def test_normalize_chunks_accepts_nested_chunk_payload_rows():
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "AAPL:0001#000",
                "document_id": "AAPL:0001",
                "text": "Revenue increased during the quarter.",
                "metadata": {
                    "source_type": "sec_10q",
                    "ticker": "AAPL",
                    "section_path": ["Part I", "Item 2"],
                },
                "context_header": "Apple Inc. (AAPL)",
                "embedding_text": "Apple Inc. (AAPL)\n\nRevenue increased during the quarter.",
            }
        ]
    )

    normalized = normalize_chunks(chunks)

    assert normalized == [
        {
            "chunk_id": "AAPL:0001#000",
            "document_id": "AAPL:0001",
            "text": "Revenue increased during the quarter.",
            "metadata": {
                "source_type": "sec_10q",
                "ticker": "AAPL",
                "section_path": ["Part I", "Item 2"],
            },
        }
    ]


def test_normalize_chunks_recovers_flattened_metadata():
    normalized = normalize_chunks(
        [
            {
                "chunk_id": "MSFT:0001#000",
                "document_id": "MSFT:0001",
                "text": "Operating income increased.",
                "metadata": None,
                "source_type": "sec_10q",
                "ticker": "MSFT",
                "form": "10-Q",
                "chunk_index": 0,
            }
        ]
    )

    assert normalized[0]["metadata"] == {
        "source_type": "sec_10q",
        "ticker": "MSFT",
        "form": "10-Q",
    }


def test_normalize_chunks_preserves_original_dict_api():
    normalized = normalize_chunks(
        [
            {
                "text": "The company amended its credit facility.",
                "metadata": {"ticker": "PEP"},
            }
        ]
    )

    assert normalized[0]["text"] == "The company amended its credit facility."
    assert normalized[0]["metadata"] == {"ticker": "PEP"}
    assert normalized[0]["chunk_id"] is None


def test_normalize_chunks_rejects_missing_text():
    with pytest.raises(ValueError, match="non-empty text"):
        normalize_chunks([{"metadata": {"ticker": "AAPL"}}])


def test_jsonl_writer_preserves_nested_metadata(tmp_path):
    output_path = tmp_path / "golden_set" / "samples.jsonl"
    dataset = pd.DataFrame(
        [
            {
                "question": "What happened to revenue?",
                "ground_truth": "Revenue increased.",
                "contexts": ["Revenue increased."],
                "metadata": {"ticker": "AAPL", "section_path": ["Part I", "Item 2"]},
                "chunk_id": "AAPL:0001#000",
                "document_id": "AAPL:0001",
            }
        ]
    )

    _write_dataset(dataset, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["contexts"] == ["Revenue increased."]
    assert saved["metadata"]["section_path"] == ["Part I", "Item 2"]
