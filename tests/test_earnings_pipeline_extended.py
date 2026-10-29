"""Extended unit tests for earnings transcript fetching and cleaning logic."""

import pytest

from src.rag.cleaning.earnings_transcript_cleaner import (
    CleanerConfig,
    clean_text,
    fiscal_quarter_label,
    normalize_speaker_role,
)
from src.rag.scraping.earnings_transcript_fetcher import (
    FetcherConfig,
    normalize_quarter,
    normalize_ticker,
)


def test_clean_text_normalizes_whitespace_and_paragraphs():
    raw = "Good morning everyone.\r\n\r\n   Welcome to the Q3 call.\n\n\n\nThank you.   "
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == "Good morning everyone.\n\nWelcome to the Q3 call.\n\nThank you."


def test_normalize_speaker_role_recognizes_standard_roles():
    assert normalize_speaker_role("Chief Executive Officer") == "CEO"
    assert normalize_speaker_role("Interim CEO") == "CEO"
    assert normalize_speaker_role("Chief Financial Officer and EVP") == "CFO"
    assert normalize_speaker_role("Conference Operator") == "operator"
    assert normalize_speaker_role("Equity Research Analyst, Morgan Stanley") == "analyst"
    assert normalize_speaker_role("Head of Investor Relations") == "investor_relations"
    assert normalize_speaker_role("Executive Vice President of Operations") == "Executive Vice President of Operations"
    assert normalize_speaker_role(None) is None
    assert normalize_speaker_role("") is None


def test_fiscal_quarter_label_converts_format():
    assert fiscal_quarter_label("2024Q1") == "2024-Q1"
    assert fiscal_quarter_label("2025q4") == "2025-Q4"


def test_fetcher_and_cleaner_config_validations():
    with pytest.raises(ValueError, match="daily_limit"):
        FetcherConfig(daily_limit=0)

    with pytest.raises(ValueError, match="min_request_interval"):
        FetcherConfig(min_request_interval=-1.0)

    with pytest.raises(ValueError, match="min_chars"):
        CleanerConfig(min_chars=0)
