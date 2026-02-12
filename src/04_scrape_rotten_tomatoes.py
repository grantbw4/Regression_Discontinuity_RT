"""
Step 4: Scrape Rotten Tomatoes scores for each movie.

For each movie from the BOM data, finds its RT page using:
  1. Direct URL slug construction (primary)
  2. RT search page fallback

Extracts: Tomatometer, Audience Score, critic/audience review counts,
genres, MPAA rating.

Input:  data/raw/bom_details.csv (or bom_index.csv if details not ready)
Output: data/raw/rt_scores.csv
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
from src.utils import (
    make_request, parse_date, setup_logging,
    load_checkpoint, save_checkpoint,
)


# --- Slug construction ---

def construct_rt_slugs(title, year=None):
    """
    Generate candidate RT URL slugs from a movie title.

    RT slug rules (observed):
    - Lowercase everything
    - Replace spaces with underscores
    - Replace '&' with 'and'
    - Remove colons, commas, periods, apostrophes, exclamation/question marks
    - Keep hyphens (some slugs use them, some replace with underscores)
    - Numbers kept as-is

    Returns list of candidate slugs, most likely first.
    """
    candidates = []

    slug = title.lower().strip()
    slug = slug.replace("&", "and")
    slug = slug.replace("'", "")
    slug = slug.replace("'", "")  # curly apostrophe
    slug = re.sub(r"[\":,\.!?;()\[\]{}]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")

    candidates.append(slug)

    # Try with hyphens replaced by underscores
    if "-" in slug:
        candidates.append(slug.replace("-", "_"))

    # Try with year suffix
    if year:
        candidates.append(f"{slug}_{year}")
        if "-" in slug:
            candidates.append(f"{slug.replace('-', '_')}_{year}")

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


# --- Score extraction ---

def extract_rt_data(html):
    """
    Extract scores, counts, genres, and rating from an RT movie page.

    Looks for embedded JSON data in the page source containing
    criticsScore, audienceScore, metadataGenres, contentRating.
    """
    result = {
        "tomatometer": None,
        "audience_score": None,
        "critic_count": None,
        "audience_count": None,
        "rt_genres": None,
        "rt_rating": None,
        "rt_title": None,
    }

    # Extract page title for validation
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    if title_tag:
        # RT titles are like "Inside Out 2 | Rotten Tomatoes"
        result["rt_title"] = title_tag.get_text(strip=True).split("|")[0].strip()

    # --- Critics score ---
    critic_match = re.search(
        r'"criticsScore"\s*:\s*\{[^}]*?"score"\s*:\s*"(\d+)"', html
    )
    if critic_match:
        result["tomatometer"] = int(critic_match.group(1))

    # --- Audience score ---
    audience_match = re.search(
        r'"audienceScore"\s*:\s*\{[^}]*?"score"\s*:\s*"(\d+)"', html
    )
    if audience_match:
        result["audience_score"] = int(audience_match.group(1))

    # --- Critic review count ---
    # Look specifically in the criticsScore block
    critic_block = re.search(r'"criticsScore"\s*:\s*(\{[^}]+\})', html)
    if critic_block:
        count_match = re.search(r'"reviewCount"\s*:\s*(\d+)', critic_block.group(1))
        if count_match:
            result["critic_count"] = int(count_match.group(1))

    # --- Audience rating count ---
    audience_block = re.search(r'"audienceScore"\s*:\s*(\{[^}]+\})', html)
    if audience_block:
        # Try ratingCount first (numeric)
        count_match = re.search(r'"ratingCount"\s*:\s*(\d+)', audience_block.group(1))
        if count_match:
            result["audience_count"] = int(count_match.group(1))
        else:
            # Try reviewCount
            count_match = re.search(r'"reviewCount"\s*:\s*(\d+)', audience_block.group(1))
            if count_match:
                result["audience_count"] = int(count_match.group(1))

    # --- Genres ---
    genres_match = re.search(r'"metadataGenres"\s*:\s*\[(.*?)\]', html)
    if genres_match:
        try:
            genres_raw = "[" + genres_match.group(1) + "]"
            genres = json.loads(genres_raw)
            result["rt_genres"] = ", ".join(genres)
        except json.JSONDecodeError:
            result["rt_genres"] = genres_match.group(1).replace('"', '').strip()

    # --- Content rating ---
    rating_match = re.search(r'"contentRating"\s*:\s*"([^"]+)"', html)
    if rating_match:
        result["rt_rating"] = rating_match.group(1)

    return result


# --- Page finding strategies ---

def try_direct_url(slug, session, headers, logger):
    """
    Try fetching an RT movie page directly by slug.
    Returns (response, url) or (None, None).
    """
    url = f"https://www.rottentomatoes.com/m/{slug}"
    resp = make_request(
        url, headers=headers,
        delay=config.RT_DELAY + random.uniform(0, 1.5),
        session=session, logger=logger,
    )
    if resp and resp.status_code == 200:
        # Verify it's actually a movie page (not a redirect to search)
        if "/m/" in resp.url and "search" not in resp.url:
            return resp, url
    return None, None


def try_search(title, year, session, headers, logger):
    """
    Search RT for a movie and return the best matching URL.
    Returns (url, match_quality) or (None, None).
    """
    search_url = f"https://www.rottentomatoes.com/search?search={quote_plus(title)}"
    resp = make_request(
        search_url, headers=headers,
        delay=config.RT_DELAY + random.uniform(0, 1.5),
        session=session, logger=logger,
    )
    if not resp or resp.status_code != 200:
        return None, None

    # Parse search results for movie links
    soup = BeautifulSoup(resp.text, "lxml")

    # Find all /m/ links in the page
    movie_links = []
    for a_tag in soup.find_all("a", href=re.compile(r"/m/[^/]+")):
        href = a_tag.get("href", "")
        link_text = a_tag.get_text(strip=True)
        if href and link_text:
            # Normalize the href
            if href.startswith("/"):
                href = f"https://www.rottentomatoes.com{href}"
            movie_links.append((href, link_text))

    # Also try extracting from embedded JS/JSON data
    # RT search pages sometimes have data in script tags
    for script in soup.find_all("script"):
        script_text = script.get_text()
        slug_matches = re.findall(r'"/m/([^"]+)"', script_text)
        for slug in slug_matches:
            url = f"https://www.rottentomatoes.com/m/{slug}"
            if url not in [ml[0] for ml in movie_links]:
                movie_links.append((url, slug.replace("_", " ")))

    if not movie_links:
        return None, None

    # Score each result against our title + year
    title_lower = title.lower().strip()
    best_url = None
    best_score = 0

    for url, link_text in movie_links:
        text_lower = link_text.lower().strip()
        # Simple scoring: exact match > contains > partial
        if text_lower == title_lower:
            score = 100
        elif title_lower in text_lower or text_lower in title_lower:
            score = 80
        else:
            # Word overlap
            title_words = set(title_lower.split())
            text_words = set(text_lower.split())
            if title_words and text_words:
                overlap = len(title_words & text_words) / max(len(title_words), len(text_words))
                score = int(overlap * 70)
            else:
                score = 0

        # Bonus for year match in URL
        if year and str(year) in url:
            score += 10

        if score > best_score:
            best_score = score
            best_url = url

    if best_url and best_score >= 50:
        return best_url, best_score
    return None, None


def get_rt_headers(session_index=0):
    """Get headers with a rotated User-Agent."""
    ua = config.RT_USER_AGENTS[session_index % len(config.RT_USER_AGENTS)]
    headers = config.HEADERS.copy()
    headers["User-Agent"] = ua
    return headers


# --- Main scraper ---

def scrape_rotten_tomatoes():
    """Main function to scrape RT scores for all movies."""
    logger = setup_logging("04_scrape_rotten_tomatoes")
    logger.info("Starting Rotten Tomatoes scrape")

    # Load BOM data to get movie list
    details_path = config.DATA_RAW / "bom_details.csv"
    index_path = config.DATA_RAW / "bom_index.csv"

    if details_path.exists():
        movies_df = pd.read_csv(details_path)
        logger.info(f"Loaded {len(movies_df)} movies from bom_details.csv")
    elif index_path.exists():
        movies_df = pd.read_csv(index_path)
        logger.info(f"Loaded {len(movies_df)} movies from bom_index.csv (details not available)")
    else:
        logger.error("No BOM data found. Run steps 01 and 02 first.")
        return

    # Check for existing checkpoint
    output_path = config.DATA_RAW / "rt_scores.csv"
    _, processed_ids = load_checkpoint(output_path)
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} already scraped")

    # Use a session for cookie persistence
    session = requests.Session()

    batch = []
    matched = 0
    unmatched = 0
    ua_index = 0

    for i, row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Scraping RT"):
        release_id = str(row["bom_release_id"])

        if release_id in processed_ids:
            continue

        title = row["title"]
        # Try to extract year from release date
        year = None
        if "release_date" in row and pd.notna(row.get("release_date")):
            date = parse_date(str(row["release_date"]))
            if date:
                year = date.year

        # Rotate user agent periodically
        headers = get_rt_headers(ua_index)
        ua_index += 1

        # Strategy 1: Direct URL construction
        slugs = construct_rt_slugs(title, year)
        found_resp = None
        found_url = None
        match_method = "unmatched"

        for slug in slugs:
            resp, url = try_direct_url(slug, session, headers, logger)
            if resp:
                found_resp = resp
                found_url = url
                match_method = "direct_url"
                break

        # Strategy 2: Search fallback
        if not found_resp:
            search_url, search_score = try_search(title, year, session, headers, logger)
            if search_url:
                # Fetch the actual movie page
                resp = make_request(
                    search_url, headers=headers,
                    delay=config.RT_DELAY + random.uniform(0, 1.5),
                    session=session, logger=logger,
                )
                if resp and resp.status_code == 200:
                    found_resp = resp
                    found_url = search_url
                    match_method = "search"

        # Extract data
        if found_resp:
            data = extract_rt_data(found_resp.text)
            data["bom_release_id"] = release_id
            data["title_searched"] = title
            data["rt_url"] = found_url
            data["match_method"] = match_method
            matched += 1
            if data["tomatometer"] is not None:
                logger.debug(
                    f"  {title}: TM={data['tomatometer']}%, "
                    f"AS={data['audience_score']}% ({match_method})"
                )
        else:
            data = {
                "bom_release_id": release_id,
                "title_searched": title,
                "rt_url": None,
                "tomatometer": None,
                "audience_score": None,
                "critic_count": None,
                "audience_count": None,
                "rt_genres": None,
                "rt_rating": None,
                "match_method": "unmatched",
                "rt_title": None,
            }
            unmatched += 1
            logger.debug(f"  {title}: UNMATCHED")

        batch.append(data)

        # Checkpoint
        if len(batch) >= config.RT_CHECKPOINT_INTERVAL:
            save_checkpoint(batch, output_path, logger)
            processed_ids.update(d["bom_release_id"] for d in batch)
            batch = []

    # Save remaining
    if batch:
        save_checkpoint(batch, output_path, logger)

    # Final summary
    total = matched + unmatched
    logger.info(
        f"Done. Matched: {matched}/{total} ({100*matched/total:.1f}%). "
        f"Unmatched: {unmatched}."
    )


if __name__ == "__main__":
    scrape_rotten_tomatoes()
