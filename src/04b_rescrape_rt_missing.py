"""
Step 4b: Re-scrape RT movies that matched a page but got empty scores.

These are movies where the direct URL resolved but RT either:
1. Served a skeleton page (bot detection) - no score JSON fields
2. Matched the wrong movie (ambiguous slug like "smile", "nobody")

Strategy: Try year-appended slugs first, use Referer header,
and fall back to search.

Input:  data/raw/rt_scores.csv
Output: Updates data/raw/rt_scores.csv in-place
"""

import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import setup_logging, parse_date

# Import extraction function from main scraper
from src import utils as _  # noqa ensure path
import importlib
scraper_mod = importlib.import_module("src.04_scrape_rotten_tomatoes")
extract_rt_data = scraper_mod.extract_rt_data
construct_rt_slugs = scraper_mod.construct_rt_slugs
try_search = scraper_mod.try_search


def make_browser_request(url, session, headers, logger, delay=2.5):
    """Make a request that looks more like a real browser."""
    time.sleep(delay + random.uniform(0.5, 2.0))

    h = headers.copy()
    h["Referer"] = "https://www.rottentomatoes.com/"
    h["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    h["Accept-Language"] = "en-US,en;q=0.9"
    h["Accept-Encoding"] = "gzip, deflate, br"
    h["Connection"] = "keep-alive"
    h["Sec-Fetch-Dest"] = "document"
    h["Sec-Fetch-Mode"] = "navigate"
    h["Sec-Fetch-Site"] = "same-origin"
    h["Sec-Fetch-User"] = "?1"
    h["Upgrade-Insecure-Requests"] = "1"
    h["Cache-Control"] = "max-age=0"

    try:
        resp = session.get(url, headers=h, timeout=30)
        if resp.status_code == 200:
            return resp
        elif resp.status_code == 403:
            logger.warning(f"HTTP 403 (blocked) for {url}")
        elif resp.status_code == 404:
            logger.debug(f"HTTP 404 for {url}")
        else:
            logger.warning(f"HTTP {resp.status_code} for {url}")
    except Exception as e:
        logger.warning(f"Request error for {url}: {e}")
    return None


def rescrape_missing():
    logger = setup_logging("04b_rescrape_rt_missing")
    logger.info("Starting RT re-scrape for movies with missing scores")

    rt_path = config.DATA_RAW / "rt_scores.csv"
    df = pd.read_csv(rt_path)

    # Also load BOM details for year info
    bom = pd.read_csv(config.DATA_RAW / "bom_details.csv")
    bom_years = dict(zip(bom["bom_release_id"].astype(str), bom["release_date"]))

    # Find movies that matched but have no scores
    mask = (
        (df["match_method"] != "unmatched")
        & df["tomatometer"].isna()
        & df["audience_score"].isna()
    )
    missing = df[mask].copy()
    logger.info(f"Found {len(missing)} movies with matched pages but no scores")

    session = requests.Session()
    # First visit RT homepage to get cookies
    homepage_headers = {
        "User-Agent": config.RT_USER_AGENTS[0],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        session.get("https://www.rottentomatoes.com", headers=homepage_headers, timeout=15)
        time.sleep(2)
    except Exception:
        pass

    fixed = 0
    still_missing = 0
    ua_index = 0

    for idx, row in tqdm(missing.iterrows(), total=len(missing), desc="Re-scraping RT"):
        release_id = str(row["bom_release_id"])
        title = row["title_searched"]

        # Get year from BOM data
        year = None
        date_str = bom_years.get(release_id)
        if date_str and pd.notna(date_str):
            date = parse_date(str(date_str))
            if date:
                year = date.year

        ua = config.RT_USER_AGENTS[ua_index % len(config.RT_USER_AGENTS)]
        ua_index += 1
        headers = {"User-Agent": ua}

        # Strategy 1: Try year-appended slug FIRST (avoids wrong-movie matches)
        slugs = construct_rt_slugs(title, year)
        # Reorder: year-appended first, then base
        year_slugs = [s for s in slugs if year and str(year) in s]
        base_slugs = [s for s in slugs if not (year and str(year) in s)]
        ordered_slugs = year_slugs + base_slugs

        found = False
        for slug in ordered_slugs:
            url = f"https://www.rottentomatoes.com/m/{slug}"
            resp = make_browser_request(url, session, headers, logger)
            if resp and "/m/" in resp.url and "search" not in resp.url:
                data = extract_rt_data(resp.text)
                if data["tomatometer"] is not None:
                    # Success - update the row
                    df.at[idx, "tomatometer"] = data["tomatometer"]
                    df.at[idx, "audience_score"] = data["audience_score"]
                    df.at[idx, "critic_count"] = data["critic_count"]
                    df.at[idx, "audience_count"] = data["audience_count"]
                    df.at[idx, "rt_genres"] = data["rt_genres"]
                    df.at[idx, "rt_rating"] = data["rt_rating"]
                    df.at[idx, "rt_title"] = data["rt_title"]
                    df.at[idx, "rt_url"] = url
                    df.at[idx, "match_method"] = "rescrape_direct"
                    fixed += 1
                    logger.info(f"  FIXED {title}: TM={data['tomatometer']}% AS={data['audience_score']}% via {slug}")
                    found = True
                    break

        if found:
            continue

        # Strategy 2: Search fallback
        search_url, search_score = try_search(title, year, session, headers, logger)
        if search_url:
            resp = make_browser_request(search_url, session, headers, logger)
            if resp:
                data = extract_rt_data(resp.text)
                if data["tomatometer"] is not None:
                    df.at[idx, "tomatometer"] = data["tomatometer"]
                    df.at[idx, "audience_score"] = data["audience_score"]
                    df.at[idx, "critic_count"] = data["critic_count"]
                    df.at[idx, "audience_count"] = data["audience_count"]
                    df.at[idx, "rt_genres"] = data["rt_genres"]
                    df.at[idx, "rt_rating"] = data["rt_rating"]
                    df.at[idx, "rt_title"] = data["rt_title"]
                    df.at[idx, "rt_url"] = search_url
                    df.at[idx, "match_method"] = "rescrape_search"
                    fixed += 1
                    logger.info(f"  FIXED {title}: TM={data['tomatometer']}% AS={data['audience_score']}% via search")
                    found = True

        if not found:
            still_missing += 1
            logger.debug(f"  STILL MISSING: {title}")

    # Save updated CSV
    df.to_csv(rt_path, index=False)
    logger.info(f"Done. Fixed: {fixed}/{len(missing)}. Still missing: {still_missing}.")
    logger.info(f"Updated {rt_path}")


if __name__ == "__main__":
    rescrape_missing()
