"""Conservative point-in-time availability timestamps for transcripts."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")
FALLBACK_DAYS_AFTER_QUARTER_END = 90


@lru_cache(maxsize=1)
def _xnys_calendar():
    return xcals.get_calendar("XNYS", start="1990-01-01", end="2050-12-31")


def _next_session_open(day: date) -> datetime:
    session = _xnys_calendar().date_to_session(pd.Timestamp(day), direction="next")
    return _xnys_calendar().session_open(session).to_pydatetime()


def next_trading_open_after_call(value: object) -> datetime | None:
    """Return the next XNYS session open after an explicit call date/time."""
    if isinstance(value, date) and not isinstance(value, datetime):
        call_day = value
    elif isinstance(value, datetime):
        call_day = value.astimezone(NEW_YORK).date() if value.tzinfo else value.date()
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            try:
                call_day = date.fromisoformat(value.strip())
            except ValueError:
                return None
        else:
            call_day = (
                parsed.astimezone(NEW_YORK).date() if parsed.tzinfo else parsed.date()
            )
    else:
        return None
    return _next_session_open(call_day + timedelta(days=1))


def quarter_fallback_open(year: int, quarter: int) -> datetime:
    """Late-side fallback: quarter end + 90 days, at or after the next XNYS open."""
    quarter_month = quarter * 3
    quarter_end = date(year, quarter_month, calendar.monthrange(year, quarter_month)[1])
    fallback_day = quarter_end + timedelta(days=FALLBACK_DAYS_AFTER_QUARTER_END)
    return _next_session_open(fallback_day)
