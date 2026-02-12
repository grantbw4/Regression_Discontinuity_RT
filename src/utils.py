"""
Shared utility functions for the data collection pipeline.
Provides HTTP request handling, text normalization, parsing, logging, and checkpointing.
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Add parent directory to path so we can import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def setup_logging(script_name):
    """Configure logging with console (INFO) and file (DEBUG) handlers."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(fh)

    logger.info(f"Logging to {log_file}")
    return logger


def make_request(url, headers=None, delay=1.5, max_retries=None, timeout=None,
                 session=None, logger=None):
    """
    HTTP GET with rate limiting, retries, and exponential backoff.

    Args:
        url: URL to fetch
        headers: HTTP headers dict (defaults to config.HEADERS)
        delay: seconds to sleep before the request
        max_retries: number of retry attempts (defaults to config.MAX_RETRIES)
        timeout: request timeout in seconds (defaults to config.REQUEST_TIMEOUT)
        session: optional requests.Session for cookie persistence
        logger: optional logger instance

    Returns:
        requests.Response or None on complete failure
    """
    if headers is None:
        headers = config.HEADERS
    if max_retries is None:
        max_retries = config.MAX_RETRIES
    if timeout is None:
        timeout = config.REQUEST_TIMEOUT
    if logger is None:
        logger = logging.getLogger(__name__)

    requester = session if session else requests

    for attempt in range(1, max_retries + 1):
        time.sleep(delay)
        try:
            resp = requester.get(url, headers=headers, timeout=timeout)
            logger.debug(f"GET {url} -> {resp.status_code}")

            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait = 60
                logger.warning(f"Rate limited (429) on {url}. Waiting {wait}s...")
                time.sleep(wait)
            elif resp.status_code == 403:
                wait = 60
                logger.warning(f"Forbidden (403) on {url}. Waiting {wait}s...")
                time.sleep(wait)
            elif resp.status_code >= 500:
                wait = config.RETRY_BACKOFF * attempt
                logger.warning(
                    f"Server error ({resp.status_code}) on {url}. "
                    f"Retry {attempt}/{max_retries} in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                return resp  # 404 etc. — don't retry

        except (requests.ConnectionError, requests.Timeout) as e:
            wait = config.RETRY_BACKOFF * attempt
            logger.warning(
                f"Request error on {url}: {e}. "
                f"Retry {attempt}/{max_retries} in {wait}s..."
            )
            time.sleep(wait)

    logger.error(f"Failed after {max_retries} retries: {url}")
    return None


def normalize_title(title):
    """
    Normalize a movie title for fuzzy matching.

    Steps:
        1. Lowercase
        2. Strip whitespace
        3. Remove parenthetical content like "(2024 Re-release)"
        4. Replace '&' with 'and'
        5. Remove punctuation (except spaces)
        6. Collapse multiple spaces
        7. Strip leading articles ('the ', 'a ', 'an ')

    Examples:
        "The Batman" -> "batman"
        "Spider-Man: No Way Home" -> "spider man no way home"
        "F9: The Fast Saga" -> "f9 fast saga"
    """
    if not title:
        return ""
    s = title.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)  # remove parenthetical content
    s = s.replace("&", "and")
    s = re.sub(r"[^\w\s]", " ", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace
    # Strip leading articles
    for article in ["the ", "a ", "an "]:
        if s.startswith(article):
            s = s[len(article):]
    return s


def parse_money(text):
    """
    Parse dollar strings into integer values.

    Handles: "$154,201,673", "$1.2B", "$400M", "-", None, empty strings.
    Returns int or None.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if text in ("-", "–", "—", "n/a", "N/A", ""):
        return None

    text = text.replace("$", "").replace(",", "").strip()

    try:
        if text.upper().endswith("B"):
            return int(float(text[:-1]) * 1_000_000_000)
        elif text.upper().endswith("M"):
            return int(float(text[:-1]) * 1_000_000)
        elif text.upper().endswith("K"):
            return int(float(text[:-1]) * 1_000)
        else:
            return int(float(text))
    except (ValueError, TypeError):
        return None


def parse_date(text, formats=None):
    """
    Parse date strings from various formats.

    Args:
        text: date string like "Jun 14, 2024" or "Dec 16, 2015"
        formats: list of strftime format strings to try

    Returns:
        datetime.date or None
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()

    if formats is None:
        formats = [
            "%b %d, %Y",   # "Jun 14, 2024"
            "%B %d, %Y",   # "June 14, 2024"
            "%Y-%m-%d",    # "2024-06-14"
            "%m/%d/%Y",    # "06/14/2024"
        ]

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def load_checkpoint(filepath):
    """
    Load existing CSV to get set of already-processed IDs.

    Args:
        filepath: Path to the checkpoint CSV

    Returns:
        (DataFrame or None, set of processed IDs)
    """
    filepath = Path(filepath)
    if filepath.exists() and filepath.stat().st_size > 0:
        try:
            df = pd.read_csv(filepath)
            if "bom_release_id" in df.columns:
                processed = set(df["bom_release_id"].dropna().astype(str))
            else:
                processed = set()
            return df, processed
        except Exception:
            return None, set()
    return None, set()


def save_checkpoint(new_rows, filepath, logger=None):
    """
    Append new rows to an existing CSV or create it.

    Args:
        new_rows: list of dicts to append
        filepath: Path to the output CSV
        logger: optional logger
    """
    if not new_rows:
        return

    filepath = Path(filepath)
    new_df = pd.DataFrame(new_rows)

    if filepath.exists() and filepath.stat().st_size > 0:
        existing = pd.read_csv(filepath)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(filepath, index=False)
    if logger:
        logger.info(f"Checkpoint saved: {len(combined)} total rows in {filepath.name}")
