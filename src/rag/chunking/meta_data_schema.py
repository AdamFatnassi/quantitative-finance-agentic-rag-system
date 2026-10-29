"""Typed, filterable metadata shared by retrieval chunks from each data source."""

from datetime import date, timezone
from typing import Literal, Optional, Union

from pydantic import AwareDatetime, BaseModel, Field, field_validator


SourceType = Literal["sec_8k", "sec_10k", "sec_10q", "financial_news", "earnings_transcript"]
SECForm = Literal["8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A"]
FiscalPeriod = Literal["Q1", "Q2", "Q3", "FY"]


class BaseChunkMetadata(BaseModel):
    """Fields shared across sources and useful for high-precision pre-filtering."""

    source_type: SourceType = Field(..., description="Dataset family; use as a retrieval filter")
    ticker: Optional[str] = Field(None, description="Normalized primary ticker, e.g. AAPL")
    cik: Optional[int] = Field(None, description="SEC Central Index Key when applicable")
    accession_number: Optional[str] = Field(None, description="SEC accession identifier when applicable")
    transcript_id: Optional[str] = Field(None, description="Transcript identifier when applicable")
    item_code: Optional[str] = Field(None, description="SEC item identifier when applicable")
    related_tickers: list[str] = Field(
        default_factory=list,
        description="Other normalized tickers explicitly associated with the source",
    )
    company_name: Optional[str] = Field(None, description="Company associated with the source")
    published_at: AwareDatetime = Field(
        ...,
        description="Timezone-aware UTC publication/SEC acceptance time for point-in-time filtering",
    )
    available_at: AwareDatetime = Field(
        ...,
        description="Timezone-aware UTC time from which this chunk is safe to use",
    )
    availability_estimated: bool = Field(
        False, description="Whether available_at uses a conservative estimated timestamp"
    )
    section_type: Optional[str] = Field(None, description="Normalized source section type")
    form_type: Optional[str] = Field(None, description="SEC form or other source form")
    source_url: Optional[str] = Field(None, description="Canonical source document URL")

    @field_validator("published_at")
    @classmethod
    def normalize_published_at_utc(
        cls, value: Optional[AwareDatetime]
    ) -> Optional[AwareDatetime]:
        return value.astimezone(timezone.utc) if value is not None else None

    @field_validator("available_at")
    @classmethod
    def normalize_available_at_utc(cls, value: AwareDatetime) -> AwareDatetime:
        return value.astimezone(timezone.utc)


class SECFilingMetadata(BaseChunkMetadata):
    """Common SEC filing fields; item/section fields locate a chunk within its filing."""

    ticker: str = Field(..., description="Normalized primary ticker, e.g. AAPL")
    company_name: str
    cik: int = Field(..., description="SEC Central Index Key")
    accession_number: str = Field(..., description="Unique SEC accession number")
    form: SECForm = Field(..., description="Exact SEC form, including amendment suffix")
    filing_date: date = Field(..., description="SEC filing date")
    report_period_end: Optional[date] = Field(
        None, description="End date of the financial period covered by the filing"
    )
    fiscal_year: Optional[int] = Field(
        None, description="Fiscal year covered; do not infer from filing date"
    )
    fiscal_period: Optional[FiscalPeriod] = Field(
        None, description="Fiscal quarter or year covered by this filing"
    )
    item_title: Optional[str] = Field(None, description="Human-readable SEC item title")
    section_path: list[str] = Field(
        default_factory=list,
        description="Hierarchical report location, e.g. ['Part II', 'Item 7A']",
    )
    part_type: Optional[Literal["item_section", "exhibit", "section", "full_filing", "fallback"]] = Field(
        None, description="Kind of filing component represented by this chunk"
    )
    exhibit_type: Optional[str] = Field(
        None, description="SEC exhibit type when the chunk comes from an exhibit"
    )
    parse_mode: Optional[str] = Field(
        None, description="Parser path used to extract this filing component"
    )
    raw_path: Optional[str] = Field(
        None, description="Workspace/data-lake-relative path to the original filing"
    )
    items_all: list[str] = Field(
        default_factory=list, description="All item codes declared for this filing"
    )
    items_matched: list[str] = Field(
        default_factory=list, description="Item codes selected for this indexed document"
    )


class SEC8KMetadata(SECFilingMetadata):
    source_type: Literal["sec_8k"] = "sec_8k"
    form: Literal["8-K", "8-K/A"]


class SEC10KMetadata(SECFilingMetadata):
    source_type: Literal["sec_10k"] = "sec_10k"
    form: Literal["10-K", "10-K/A"]
    fiscal_period: Optional[Literal["FY"]] = None


class SEC10QMetadata(SECFilingMetadata):
    source_type: Literal["sec_10q"] = "sec_10q"
    form: Literal["10-Q", "10-Q/A"]
    fiscal_period: Optional[Literal["Q1", "Q2", "Q3"]] = None

# We're leaving it for now, we're going to focus only on the other sources 
class NewsMetadata(BaseChunkMetadata):
    source_type: Literal["financial_news"] = "financial_news"
    headline: str = Field(..., description="Article headline")
    publisher: str = Field(..., description="Publication or news provider")
    article_url: Optional[str] = Field(None, description="Canonical article URL")
    category: Optional[str] = Field(
        None, description="Normalized topic/category for exact filtering"
    )


class TranscriptMetadata(BaseChunkMetadata):
    source_type: Literal["earnings_transcript"] = "earnings_transcript"
    ticker: str = Field(..., description="Normalized primary ticker, e.g. AAPL")
    cik: int = Field(..., description="SEC Central Index Key for the transcript company")
    transcript_id: str = Field(..., description="Stable identifier for the source transcript")
    fiscal_quarter: str = Field(..., description="Fiscal call period, e.g. 2024-Q4")
    published_at: Optional[AwareDatetime] = Field(
        None,
        description="Actual call publication timestamp when supplied by the provider; never inferred",
    )
    retrieved_at: Optional[AwareDatetime] = Field(
        None, description="UTC time this transcript was acquired"
    )
    available_at: AwareDatetime = Field(
        ...,
        description="Conservative timestamp for when this system can use the transcript",
    )
    availability_estimated: bool
    availability_basis: Literal[
        "provider_published_at",
        "call_datetime_lag",
        "trusted_call_date_lag",
        "quarter_end_plus_90_days",
    ] = Field(
        ...,
        description="Timestamp source used to determine conservative availability",
    )
    raw_path: Optional[str] = Field(None, description="Data-lake path to the original API response")
    section: Optional[Literal["prepared_remarks", "q_and_a"]] = Field(
        None, description="Transcript segment; absent when the source is not split by segment"
    )
    speaker_name: Optional[str] = Field(None, description="Speaker for this transcript chunk")
    speaker_role: Optional[str] = Field(
        None, description="Normalized role, e.g. CEO, CFO, analyst, or operator"
    )
    turn_index: Optional[int] = Field(None, ge=0, description="Zero-based turn in the transcript")
    sentiment: Optional[str] = Field(None, description="Provider-supplied turn sentiment")


ChunkMetadata = Union[
    SEC8KMetadata,
    SEC10KMetadata,
    SEC10QMetadata,
    NewsMetadata,
    TranscriptMetadata,
]


class ChunkPayload(BaseModel):
    """A retrieval unit; metadata carries exact filters while text carries semantics."""

    chunk_id: str = Field(..., description="Stable unique identifier for this chunk")
    document_id: str = Field(..., description="Identifier shared by chunks from one document")
    text: str = Field(..., description="Chunk text shown to the user and used for citations")
    word_count: int = Field(..., ge=0, description="Number of words in text")
    chunk_index: int = Field(..., ge=0, description="Zero-based position within the document")
    total_chunks: int = Field(..., gt=0, description="Number of chunks in the document")
    metadata: ChunkMetadata = Field(..., discriminator="source_type")
