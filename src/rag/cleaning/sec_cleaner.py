"""
sec_cleaner.py - turn raw SEC filings into clean, section-tagged rows for chunking.

Responsibility: SELECT and CLEAN. It reads what sec_8k_scraper.py stored and does
NOT touch the network. It does not chunk or embed (that is the next stage).

Pipeline
--------
1. Load the scraper manifest.
2. Point-in-time window: keep filings accepted by SEC in
       (cutoff - lookback_days, cutoff],  cutoff = market close (16:00 New York)
   of `as_of`. Nothing published after the cutoff can leak into a signal's context.
3. Keep 8-K filings with a target item; keep 10-K / 10-Q reports.
4. Parse the complete submission file into its documents (main 8-K + exhibits).
5. Clean each kept document:
     * HTML -> plain text (hidden inline-XBRL data, scripts and styles dropped)
     * Unicode / whitespace normalisation, table rows kept readable
     * 8-K: split per item, keep configured target items and EX-99.x exhibits
     * 10-K / 10-Q: split by Part and Item; preserve the complete filing as a
       fallback when the report headings cannot be identified
6. Write one row per report section, 8-K item, or exhibit.

Output: <lake_root>/processed/8k-sec/filings_clean.parquet
Rows include SEC identity, dates, form, section path, item/exhibit identifiers,
source URL, raw path, and cleaned text.
`published_at` is the SEC acceptance time (UTC): use it as the date payload in the
vector store for point-in-time filtering.

Usage
-----
    python -m src.rag.cleaning.sec_cleaner --tickers AAPL MSFT --as-of 2026-09-30
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

# Must match sec_8k_scraper.py
RAW_DIRNAME = "8k-sec"
MANIFEST_NAME = "manifest.jsonl"

# ---------------------------------------------------------------------------
# What to keep
# ---------------------------------------------------------------------------

TARGET_ITEMS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.02": "Departure/Election of Directors or Officers; Compensatory Arrangements",
    # "4.01": "Changes in Registrant's Certifying Accountant",   # auditor resignations/dismissals
}

MAIN_DOC_TYPES = ("8-K", "8-K/A")
KEEP_EXHIBIT_PREFIXES = ("EX-99",)          # press releases / investor materials
SUPPORTED_FORMS = ("8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A")
MAIN_FORMS = {
    "8-K": ("8-K", "8-K/A"),
    "10-K": ("10-K", "10-K/A"),
    "10-Q": ("10-Q", "10-Q/A"),
}
SOURCE_TYPES = {
    "8-K": "sec_8k", "8-K/A": "sec_8k",
    "10-K": "sec_10k", "10-K/A": "sec_10k",
    "10-Q": "sec_10q", "10-Q/A": "sec_10q",
}

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)


@dataclass(frozen=True)
class CleanerConfig:
    lake_root: Path = Path("data/lake")
    target_items: dict = field(default_factory=lambda: dict(TARGET_ITEMS))
    keep_exhibit_prefixes: tuple = KEEP_EXHIBIT_PREFIXES
    forms: tuple[str, ...] = SUPPORTED_FORMS
    min_chars: int = 40                      # drop empty / near-empty parts


# ---------------------------------------------------------------------------
# Step 1-3: manifest, point-in-time window, item selection
# ---------------------------------------------------------------------------

def pit_cutoff(as_of: date) -> pd.Timestamp:
    """Market close (16:00 New York) of `as_of`, in UTC. Matches 'known at the close of day t'."""
    close_et = datetime.combine(as_of, MARKET_CLOSE, tzinfo=MARKET_TZ)
    return pd.Timestamp(close_et).tz_convert("UTC")


def load_manifest(lake_root: Path) -> pd.DataFrame:
    path = lake_root / "raw" / RAW_DIRNAME / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run sec_8k_scraper.py first.")
    with path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if not records:
        return pd.DataFrame(columns=[
            "ticker", "cik", "company_name", "accession_number", "form", "items",
            "filing_date", "report_date", "acceptance_datetime", "source_url", "raw_path",
            "published_at", "available_at", "availability_estimated",
        ])
    df = pd.DataFrame(records)
    if "acceptance_datetime" not in df:
        raise ValueError("SEC manifest is missing acceptance_datetime")
    accepted = pd.to_datetime(df["acceptance_datetime"], utc=True, errors="coerce")
    if accepted.isna().any():
        missing = int(accepted.isna().sum())
        raise ValueError(
            f"{missing} SEC manifest record(s) lack a valid EDGAR acceptance datetime"
        )
    df["published_at"] = accepted
    df["available_at"] = accepted
    df["availability_estimated"] = False
    return df


def parse_items(items: object) -> list[str]:
    if not isinstance(items, str):
        return []
    return [item.strip() for item in items.split(",") if item.strip()]


def select_filings(manifest: pd.DataFrame, target_items: dict,
                   tickers: Optional[list[str]] = None,
                   as_of: Optional[date] = None, lookback_days: int = 90,
                   forms: tuple[str, ...] = SUPPORTED_FORMS,
                   start_date: date | None = None,
                   end_date: date | None = None) -> pd.DataFrame:
    if manifest.empty:
        return manifest.assign(items_all=pd.Series(dtype=object),
                               items_matched=pd.Series(dtype=object))
    df = manifest
    df = df[df["form"].isin(forms)].copy()
    if tickers:
        wanted = {t.strip().upper().replace(".", "-") for t in tickers}
        df = df[df["ticker"].isin(wanted)]
    if (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be provided together")
    if start_date is not None and end_date is not None:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        start_cutoff = pd.Timestamp(
            datetime.combine(start_date, time.min, tzinfo=MARKET_TZ)
        ).tz_convert("UTC")
        end_cutoff = pit_cutoff(end_date)
        df = df[(df["published_at"] >= start_cutoff) & (df["published_at"] <= end_cutoff)]
    elif as_of is not None:
        cutoff = pit_cutoff(as_of)
        df = df[(df["published_at"] <= cutoff) & (df["published_at"] > cutoff - pd.Timedelta(days=lookback_days))]

    df["items_all"] = df.get("items", pd.Series("", index=df.index)).map(parse_items)
    df["items_matched"] = df["items_all"].map(
        lambda codes: [code for code in codes if code in target_items]
    )
    is_8k = df["form"].astype(str).str.startswith("8-K")
    keep = ~is_8k | df["items_matched"].map(bool)
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: split the complete submission file into documents
# ---------------------------------------------------------------------------

@dataclass
class SubmissionDoc:
    doc_type: str
    filename: str
    text: str


_DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S)
_TEXT_RE = re.compile(r"<TEXT>(.*)</TEXT>", re.S)
_TYPE_RE = re.compile(r"^<TYPE>(.+)$", re.M)
_FILENAME_RE = re.compile(r"^<FILENAME>(.+)$", re.M)


def parse_submission(raw: str) -> list[SubmissionDoc]:
    docs = []
    for block in _DOCUMENT_RE.findall(raw):
        text_match = _TEXT_RE.search(block)
        type_match = _TYPE_RE.search(block)
        if not text_match or not type_match:
            continue
        filename = _FILENAME_RE.search(block)
        docs.append(SubmissionDoc(type_match.group(1).strip(), filename.group(1).strip() if filename else "",
                                  text_match.group(1)))
    return docs


# ---------------------------------------------------------------------------
# Step 5a: text extraction and normalisation
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """HTML -> text using only the standard library."""

    _SKIP_TAGS = {"script", "style", "title", "ix:header"}      # ix:header = hidden XBRL facts
    _VOID_TAGS = {"br", "img", "hr", "meta", "input", "link"}
    _BLOCK_TAGS = {"p", "div", "br", "tr", "li", "ul", "ol", "table", "hr", "section",
                   "h1", "h2", "h3", "h4", "h5", "h6"}
    _CELL_TAGS = {"td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_tag: Optional[str] = None
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        hidden = "display:none" in style and tag not in self._VOID_TAGS
        if tag in self._SKIP_TAGS or hidden:
            self._skip_tag, self._skip_depth = tag, 1
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")
        elif tag in self._CELL_TAGS:
            self._parts.append(" | ")

    def handle_data(self, data):
        if not self._skip_tag:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


_HTML_HINT = re.compile(r"<\s*(html|body|div|p|table|font|span|br)\b", re.I)
_BINARY_HINT = re.compile(r"^(<PDF>|begin \d{3} )")


def is_binary_payload(text: str) -> bool:
    """PDF / uuencoded attachments embedded in the submission file: not readable text."""
    return bool(_BINARY_HINT.match(text.lstrip()[:200]))


def to_plain_text(raw_text: str) -> str:
    if _HTML_HINT.search(raw_text[:20000]):
        extractor = _TextExtractor()
        extractor.feed(raw_text)
        extractor.close()
        raw_text = extractor.text()
    return normalize_text(raw_text)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r", "\n")      # NFKC also fixes non-breaking spaces
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", line)
        line = re.sub(r"(?:\|\s*){2,}", "| ", line)                      # runs of empty table cells
        line = re.sub(r"^[|\s]+|[|\s]+$", "", line)                      # dangling pipes at the edges
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ---------------------------------------------------------------------------
# Step 5b: filing body -> form-specific sections
# ---------------------------------------------------------------------------

_ITEM_HEADER = re.compile(r"^[ \t]*item[ \t]+(\d{1,2}\.\d{2})\b\.?[ \t]*(.*)$", re.I | re.M)
_SIGNATURE = re.compile(r"^[ \t]*signatures?[ \t]*$", re.I | re.M)


def extract_item_sections(body: str) -> dict[str, str]:
    """
    Split the main 8-K text into {item_code: text}. Everything before the first
    'Item x.xx' header (cover page) and from 'SIGNATURE' on is discarded.
    Returns {} when no item headers are found.
    """
    headers = list(_ITEM_HEADER.finditer(body))
    if not headers:
        return {}
    signature = _SIGNATURE.search(body, headers[-1].start())
    body_end = signature.start() if signature else len(body)

    sections: dict[str, list[str]] = {}
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else body_end
        text = normalize_text(header.group(2) + "\n" + body[header.end():end])
        sections.setdefault(header.group(1), []).append(text)
    return {code: "\n\n".join(parts).strip() for code, parts in sections.items()}


_REPORT_HEADING = re.compile(
    r"^[ \t]*(?:(PART[ \t]+[IVX]+)\b[ \t]*[-—:]?[ \t]*(.*)|"
    r"ITEM[ \t]+(\d{1,2}[A-Z]?)\b\.?[ \t]*(.*))$",
    re.I | re.M,
)


def extract_report_sections(body: str) -> list[dict[str, str]]:
    """Split 10-K / 10-Q text at line-level Part and Item headings."""
    matches = list(_REPORT_HEADING.finditer(body))
    item_matches = [match for match in matches if match.group(3)]
    if not item_matches:
        return []

    signature = _SIGNATURE.search(body, item_matches[-1].start())
    body_end = signature.start() if signature else len(body)
    sections: list[dict[str, str]] = []
    current_part = ""

    for index, match in enumerate(matches):
        if match.start() >= body_end:
            break
        if match.group(1):
            current_part = f"{match.group(1).upper()} {match.group(2).strip()}".strip()
            continue

        code = match.group(3).upper()
        title = match.group(4).strip().rstrip(".")
        end = next(
            (next_match.start() for next_match in matches[index + 1:] if next_match.start() < body_end),
            body_end,
        )
        text = normalize_text(f"Item {code} {title}\n{body[match.end():end]}")
        path = " > ".join(part for part in (current_part, f"Item {code}") if part)
        sections.append({
            "item_code": code,
            "item_label": title or f"Item {code}",
            "section_path": path,
            "text": text,
        })
    return sections


# ---------------------------------------------------------------------------
# Step 5c: one filing -> output rows
# ---------------------------------------------------------------------------

def clean_filing(record: pd.Series, raw_text: str, cfg: CleanerConfig) -> list[dict]:
    form = str(record["form"])
    source_type = SOURCE_TYPES.get(form)
    if source_type is None:
        logger.warning("unsupported SEC form %s; skipped", form)
        return []

    report_date = record.get("report_date")
    report_period_end = (
        str(report_date) if isinstance(report_date, (date, datetime, pd.Timestamp))
        else report_date if isinstance(report_date, str) and report_date else None
    )
    try:
        report_year = date.fromisoformat(report_period_end).year if report_period_end else None
    except ValueError:
        report_year = None

    base = {
        "ticker": record["ticker"], "cik": record["cik"], "company_name": record["company_name"],
        "accession_number": record["accession_number"], "transcript_id": None,
        "form": form, "form_type": form,
        "source_type": source_type,
        "filing_date": record["filing_date"],
        "report_period_end": report_period_end, "fiscal_year": report_year,
        "fiscal_period": "FY" if form.startswith("10-K") else None,
        "published_at": record["published_at"].isoformat(),
        "available_at": record["available_at"].isoformat(),
        "availability_estimated": False,
        "items_all": record["items_all"], "items_matched": record["items_matched"],
        "source_url": record["source_url"], "raw_path": record["raw_path"],
    }
    rows: list[dict] = []
    seen_parts: dict[str, int] = {}

    def add(part_type: str, text: str, label: str, item_code: Optional[str] = None,
            exhibit_type: Optional[str] = None, section_path: Optional[list[str]] = None,
            parse_mode: str = "sections") -> None:
        if len(text) < cfg.min_chars:
            return
        location = section_path or [label]
        part = re.sub(r"[^A-Za-z0-9.-]+", "-", "-".join(location)).strip("-")
        count = seen_parts.get(part, 0)
        seen_parts[part] = count + 1
        part_key = f"{part}-{count:02d}" if count > 0 else part
        rows.append({**base,
                     "doc_id": f"{base['accession_number']}:{part_key}",
                     "part_type": part_type, "item_code": item_code, "item_label": label,
                     "section_path": location, "exhibit_type": exhibit_type,
                     "parse_mode": parse_mode,
                     "text": text, "n_chars": len(text)})

    for doc in parse_submission(raw_text):
        if is_binary_payload(doc.text):
            continue

        if doc.doc_type in MAIN_FORMS.get(form.split("/")[0], ()):
            text = to_plain_text(doc.text)
            if form.startswith("8-K"):
                sections = extract_item_sections(text)
                if sections:
                    for code in record["items_matched"]:
                        if code in sections:
                            add("item_section", sections[code], cfg.target_items[code],
                                item_code=code, section_path=[f"Item {code}"])
                        else:
                            logger.debug(
                                "%s: item %s listed in SEC metadata but no section found",
                                record["accession_number"], code,
                            )
                else:
                    add("full_filing", text, "Full 8-K body (item sections not detected)",
                        section_path=["Full filing"], parse_mode="fallback")
            else:
                sections = extract_report_sections(text)
                if sections:
                    for section in sections:
                        add(
                            "section", section["text"], section["item_label"],
                            item_code=section["item_code"],
                            section_path=section["section_path"].split(" > "),
                        )
                else:
                    add("full_filing", text, f"Full {form} filing",
                        section_path=["Full filing"], parse_mode="fallback")

        elif (form.startswith("8-K")
              and doc.doc_type.startswith(cfg.keep_exhibit_prefixes)):
            add("exhibit", to_plain_text(doc.text), f"Exhibit {doc.doc_type}",
                exhibit_type=doc.doc_type, section_path=[f"Exhibit {doc.doc_type}"])

    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class SecFilingCleaner:
    def __init__(self, config: CleanerConfig = CleanerConfig()):
        self.cfg = config
        self.out_path = config.lake_root / "processed" / RAW_DIRNAME / "filings_clean.parquet"

    def run(self, tickers: Optional[list[str]] = None, as_of: Optional[date] = None,
            lookback_days: int = 90, write: bool = True,
            start_date: date | None = None,
            end_date: date | None = None) -> pd.DataFrame:
        manifest = load_manifest(self.cfg.lake_root)
        selected = select_filings(
            manifest, self.cfg.target_items, tickers, as_of, lookback_days,
            self.cfg.forms, start_date, end_date
        )
        logger.info("%d of %d supported filings selected", len(selected), len(manifest))

        rows: list[dict] = []
        for _, record in selected.iterrows():
            raw_file = self.cfg.lake_root / record["raw_path"]
            if not raw_file.exists():
                logger.warning("raw file missing, skipped: %s", raw_file)
                continue
            raw_text = raw_file.read_bytes().decode("utf-8", errors="replace")
            rows.extend(clean_filing(record, raw_text, self.cfg))

        columns = [
            "doc_id", "source_type", "ticker", "cik", "company_name", "accession_number",
            "transcript_id", "form", "form_type", "filing_date", "report_period_end",
            "fiscal_year", "fiscal_period",
            "published_at", "available_at", "availability_estimated",
            "items_all", "items_matched", "part_type", "section_type", "item_code",
            "item_label", "section_path", "exhibit_type", "parse_mode", "text", "n_chars",
            "source_url", "raw_path",
        ]
        df = pd.DataFrame(rows, columns=columns)
        if not df.empty:
            df["section_type"] = df["part_type"]
            df = df.sort_values(["ticker", "published_at", "doc_id"]).reset_index(drop=True)
        if write:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.out_path, index=False)
            logger.info("wrote %d rows -> %s", len(df), self.out_path)
        return df


# Retain the original class name for existing 8-K callers.
Sec8KCleaner = SecFilingCleaner


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean SEC filings from the data lake.")
    parser.add_argument("--tickers", nargs="+", default=None, help="default: every ticker in the manifest")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD point-in-time cutoff (default: no cutoff)")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--lake-root", default="data/lake")
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--forms", nargs="+", choices=SUPPORTED_FORMS, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = date.fromisoformat(args.as_of) if args.as_of else None

    cfg = CleanerConfig(
        lake_root=Path(args.lake_root), min_chars=args.min_chars,
        forms=tuple(args.forms) if args.forms else SUPPORTED_FORMS,
    )
    df = SecFilingCleaner(cfg).run(args.tickers, as_of, args.lookback_days)

    if df.empty:
        print("No rows produced for the requested forms and date window.")
        return
    print(f"\n{len(df)} rows from {df['accession_number'].nunique()} filings")
    summary = (df.assign(part=df["section_path"].map(
                    lambda path: " > ".join(path) if isinstance(path, list) else str(path)))
                 .groupby(["ticker", "form", "part"])
                 .agg(rows=("doc_id", "size"), chars=("n_chars", "sum")))
    print(summary.to_string())


if __name__ == "__main__":
    main()
