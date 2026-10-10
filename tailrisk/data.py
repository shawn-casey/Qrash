"""Read the value-weighted daily table in Kenneth French's industry file."""

import hashlib
import io
import re
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "12_Industry_Portfolios_daily_CSV.zip"
)


def parse_daily(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "Average Value Weighted Returns" in line)
    header = next(i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith(","))
    rows = []
    for line in lines[header + 1 :]:
        if not re.match(r"^\s*\d{8},", line):
            if rows:
                break
            continue
        rows.append(line)
    if not rows:
        raise ValueError("No daily value-weighted returns found")
    frame = pd.read_csv(io.StringIO("\n".join([lines[header], *rows])), index_col=0)
    frame.columns = frame.columns.str.strip()
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    frame = frame.replace([-99.99, -999], np.nan) / 100.0
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    return frame


def fetch_daily(cache: Path) -> tuple[pd.DataFrame, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "12_Industry_Portfolios_daily_CSV.zip"
    if not archive.exists():
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive.write_bytes(response.read())
    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        name = next(name for name in zipped.namelist() if name.lower().endswith(".csv"))
        text = zipped.read(name).decode("utf-8-sig")
    provenance = {
        "url": DATA_URL,
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "archive_modified_at_utc": datetime.fromtimestamp(archive.stat().st_mtime, UTC).isoformat(),
        "table": "Average Value Weighted Returns -- Daily",
        "units": "decimal simple returns (source percentages divided by 100)",
        "source_header": text.splitlines()[:8],
    }
    return parse_daily(text), provenance


def holding_period_losses(daily: pd.DataFrame, days: int = 10) -> pd.DataFrame:
    if days < 1:
        raise ValueError("days must be positive")
    # Missing observations invalidate the complete window; do not join across missing days.
    compounded = (1.0 + daily).rolling(days, min_periods=days).apply(np.prod, raw=True) - 1.0
    return -compounded.dropna()
