"""Check PostgreSQL connectivity using DATABASE_URL without displaying credentials."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url


ROOT = Path(__file__).resolve().parents[1]


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without requiring python-dotenv."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _connection_url() -> URL | None:
    raw_url = os.environ.get("DATABASE_URL") or _read_dotenv(ROOT / ".env").get("DATABASE_URL")
    if not raw_url:
        return None
    return make_url(raw_url)


def main() -> int:
    try:
        url = _connection_url()
    except Exception as exc:
        print(f"DB connection test: FAIL (invalid DATABASE_URL: {type(exc).__name__})")
        return 2

    if url is None:
        print("DB connection test: FAIL (DATABASE_URL not found in environment or project .env)")
        return 2

    safe_url = url.set(username=None, password=None, query={})
    print(f"Testing PostgreSQL target: {safe_url}")

    engine = None
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 5})
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar_one()
        if result != 1:
            raise RuntimeError("SELECT 1 returned an unexpected result")
        print("DB connection test: PASS (authenticated and SELECT 1 succeeded)")
        return 0
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        print(f"DB connection test: FAIL (Python database driver unavailable: {missing})")
        print("Install the driver required by DATABASE_URL in the project environment, then retry.")
        return 1
    except Exception as exc:
        print(f"DB connection test: FAIL ({type(exc).__name__})")
        print("Check that PostgreSQL is running, the host/port are reachable, and credentials/database are correct.")
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
