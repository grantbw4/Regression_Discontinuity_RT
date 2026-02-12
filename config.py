"""
Centralized configuration for the Rotten Tomatoes RDD data collection pipeline.
All paths, study parameters, rate limits, and HTTP settings in one place.
"""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

# === Study Parameters ===
START_DATE = "2021-09-01"
END_DATE = "2026-02-07"
MIN_OPENING_THEATERS = 600
RD_THRESHOLD = 60  # Fresh/Rotten cutoff for both critic and audience scores
BOM_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

# Movies released within this many days of END_DATE are flagged as in_theaters
IN_THEATERS_WINDOW_DAYS = 56  # 8 weeks

# === Rate Limiting (seconds between requests) ===
BOM_DELAY = 1.5
RT_DELAY = 2.0
TN_DELAY = 1.0

# === HTTP Headers ===
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

# Rotating User-Agents for RT (which does bot detection)
RT_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

# === Retry Settings ===
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds, multiplied by attempt number
REQUEST_TIMEOUT = 30  # seconds

# === Checkpoint Settings ===
BOM_DETAILS_CHECKPOINT_INTERVAL = 50
RT_CHECKPOINT_INTERVAL = 25

# === Fuzzy Matching Settings ===
FUZZY_MATCH_ACCEPT_THRESHOLD = 85
FUZZY_MATCH_REVIEW_THRESHOLD = 70
