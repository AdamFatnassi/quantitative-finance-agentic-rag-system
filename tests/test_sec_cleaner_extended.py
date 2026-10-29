"""Unit tests for SEC cleaner HTML stripping, section extraction, and point-in-time filtering."""

from datetime import date
import json
import pandas as pd

from src.rag.cleaning.sec_cleaner import (
    CleanerConfig,
    _TextExtractor,
    extract_item_sections,
    normalize_text,
    load_manifest,
    parse_submission,
    pit_cutoff,
    select_filings,
    to_plain_text,
    TARGET_ITEMS,
)


def test_sec_available_at_uses_acceptance_datetime_only(tmp_path):
    manifest_path = tmp_path / "raw" / "8k-sec" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "ticker": "AAPL",
        "form": "8-K",
        "filing_date": "2024-05-01",
        "acceptance_datetime": "2024-05-01T15:30:00-04:00",
    }) + "\n", encoding="utf-8")

    manifest = load_manifest(tmp_path)

    expected = pd.Timestamp("2024-05-01T19:30:00Z")
    assert manifest.iloc[0]["available_at"] == expected
    assert manifest.iloc[0]["published_at"] == expected
    assert not bool(manifest.iloc[0]["availability_estimated"])

    manifest_path.write_text(json.dumps({
        "ticker": "AAPL",
        "form": "8-K",
        "filing_date": "2024-05-01",
        "acceptance_datetime": None,
    }) + "\n", encoding="utf-8")
    try:
        load_manifest(tmp_path)
    except ValueError as exc:
        assert "lack a valid EDGAR acceptance datetime" in str(exc)
    else:
        raise AssertionError("SEC filing without acceptance time was accepted")


def test_eastern_naive_datetime_converts_to_correct_utc_across_dst_boundary():
    from datetime import datetime
    from src.rag.scraping.sec_scraper import normalize_acceptance_datetime

    # Winter: Eastern Standard Time (EST = UTC-5)
    est_naive = datetime(2024, 1, 15, 16, 30, 0)
    utc_est = normalize_acceptance_datetime(est_naive)
    assert utc_est == "2024-01-15T21:30:00+00:00"

    # Summer: Eastern Daylight Time (EDT = UTC-4)
    edt_naive = datetime(2024, 6, 15, 16, 30, 0)
    utc_edt = normalize_acceptance_datetime(edt_naive)
    assert utc_edt == "2024-06-15T20:30:00+00:00"

    # String representations across DST boundaries
    assert normalize_acceptance_datetime("2024-03-08 17:00:00") == "2024-03-08T22:00:00+00:00"  # EST (UTC-5)
    assert normalize_acceptance_datetime("2024-03-12 17:00:00") == "2024-03-12T21:00:00+00:00"  # EDT (UTC-4)


def test_html_stripping_drops_scripts_styles_and_hidden_elements():
    html = """
    <html>
      <head>
        <style>body { font-size: 12px; }</style>
        <script>console.log("drop me");</script>
      </head>
      <body>
        <div style="display:none">Hidden XBRL data</div>
        <p>Visible executive summary paragraph.</p>
        <table>
          <tr><td>Revenue</td><td>$100M</td></tr>
        </table>
      </body>
    </html>
    """
    clean = to_plain_text(html)
    assert "drop me" not in clean
    assert "font-size" not in clean
    assert "Hidden XBRL data" not in clean
    assert "Visible executive summary paragraph." in clean
    assert "Revenue | $100M" in clean


def test_normalize_text_cleans_excess_pipes_and_whitespace():
    raw = "   Some text  with   spaces.  \n\n\n\n| | Col 1 | Col 2 | | \n\n"
    norm = normalize_text(raw)
    assert "\n\n\n" not in norm
    assert "  with   spaces" not in norm
    assert "Col 1 | Col 2" in norm


import textwrap

def test_parse_submission_extracts_documents():
    submission_raw = textwrap.dedent("""
    <DOCUMENT>
    <TYPE>8-K
    <FILENAME>filing.htm
    <TEXT>
    Item 2.02 Results of Operations.
    Revenue was strong.
    </TEXT>
    </DOCUMENT>
    <DOCUMENT>
    <TYPE>EX-99.1
    <FILENAME>exhibit991.htm
    <TEXT>
    Press release details.
    </TEXT>
    </DOCUMENT>
    """).strip()
    docs = parse_submission(submission_raw)
    assert len(docs) == 2
    assert docs[0].doc_type == "8-K"
    assert docs[0].filename == "filing.htm"
    assert "Item 2.02" in docs[0].text
    assert docs[1].doc_type == "EX-99.1"
    assert "Press release" in docs[1].text


def test_extract_item_sections_8k():
    body = """
    UNITED STATES SECURITIES AND EXCHANGE COMMISSION
    Item 1.01 Entry into a Material Definitive Agreement.
    The Company entered into a revolving credit facility.
    Item 2.02 Results of Operations and Financial Condition.
    Operating margin reached 30%.
    SIGNATURE
    Pursuant to the requirements of the Securities Exchange Act...
    """
    items = extract_item_sections(body)
    assert "1.01" in items
    assert "2.02" in items
    assert "revolving credit facility" in items["1.01"]
    assert "Operating margin reached 30%" in items["2.02"]
    assert "Pursuant to the requirements" not in items["2.02"]


def test_point_in_time_filtering_excludes_after_cutoff():
    # As of 2024-05-01, cutoff is 16:00 ET = 20:00 UTC
    as_of = date(2024, 5, 1)
    cutoff = pit_cutoff(as_of)
    assert cutoff == pd.Timestamp("2024-05-01 20:00:00+00:00")

    manifest = pd.DataFrame([
        {
            "ticker": "AAPL",
            "form": "8-K",
            "items": "2.02",
            # Published before cutoff: 19:30 UTC
            "published_at": pd.Timestamp("2024-05-01 19:30:00+00:00"),
        },
        {
            "ticker": "AAPL",
            "form": "8-K",
            "items": "2.02",
            # Published after cutoff: 20:30 UTC (e.g. after hours)
            "published_at": pd.Timestamp("2024-05-01 20:30:00+00:00"),
        },
    ])

    selected = select_filings(manifest, TARGET_ITEMS, tickers=["AAPL"], as_of=as_of, lookback_days=10)
    assert len(selected) == 1
    assert selected.iloc[0]["published_at"] == pd.Timestamp("2024-05-01 19:30:00+00:00")
