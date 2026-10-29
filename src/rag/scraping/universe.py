"""Read the filtered stat-arb universe snapshot used for text ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

UNIVERSE_COLUMNS = ("ticker", "cik", "sector", "first_date", "last_date")


def universe_path(lake_root: Path) -> Path:
    """Resolve data/universe.parquet for a lake rooted at data/lake."""
    return lake_root.parent / "universe.parquet"


def load_universe(lake_root: Path) -> pd.DataFrame:
    path = universe_path(lake_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run the quantitative pipeline to create the filtered universe."
        )
    universe = pd.read_parquet(path)
    missing = set(UNIVERSE_COLUMNS) - set(universe.columns)
    if missing:
        raise ValueError(f"{path} is missing universe columns: {sorted(missing)}")
    rows = universe[list(UNIVERSE_COLUMNS)].copy()
    rows["ticker"] = rows["ticker"].astype(str).str.strip().str.upper()
    cik_values = rows["cik"].astype(str).str.replace(r"\.0$", "", regex=True)
    rows["cik"] = cik_values.str.extract(r"(\d+)")[0].str.zfill(10)
    if rows["ticker"].duplicated().any():
        raise ValueError(f"{path} contains duplicate tickers")
    if rows["cik"].isna().any() or not rows["cik"].str.fullmatch(r"\d{10}").all():
        raise ValueError(f"{path} contains invalid CIK values")
    if rows["ticker"].eq("").any():
        raise ValueError(f"{path} contains an empty ticker")
    return rows


def ingestion_tickers(lake_root: Path, tickers: list[str] | None = None) -> list[str]:
    """Use an explicit test override or every ticker in the filtered universe."""
    if tickers is not None:
        return list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
    return load_universe(lake_root)["ticker"].tolist()
