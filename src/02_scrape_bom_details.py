"""
Step 2: Scrape individual Box Office Mojo release pages.

For each movie in the BOM index, fetches the detailed release page to get:
- Opening weekend gross and theaters
- Widest release theaters
- Total domestic gross
- MPAA rating, genres, release date, distributor

Supports checkpoint/resume for the ~890 requests.

Input:  data/raw/bom_index.csv
Output: data/raw/bom_details.csv
"""

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import (
    make_request, parse_money, setup_logging,
    load_checkpoint, save_checkpoint,
)


def get_summary_div_text(soup, label_text):
    """
    Find a BOM summary div by its label span and return the parent div's
    pipe-separated text parts (excluding the label itself).

    BOM page structure: <div class="a-section a-spacing-none">
                           <span>Label</span>
                           <span>Value content</span>
                        </div>

    Returns list of text parts after the label, or empty list.
    """
    # Find span with exact label text
    label = soup.find("span", string=re.compile(rf"^\s*{re.escape(label_text)}\s*$"))
    if not label:
        # Try partial match (e.g., "Domestic (" starts with "Domestic")
        for span in soup.find_all("span"):
            if span.get_text(strip=True).startswith(label_text):
                label = span
                break
    if not label:
        return []

    # Get the parent div's full text, split by pipe separator
    parent = label.parent
    if not parent:
        return []

    parts = parent.get_text(separator="|", strip=True).split("|")
    # Find the label part and return everything after it
    for i, part in enumerate(parts):
        if part.strip().startswith(label_text):
            return [p.strip() for p in parts[i + 1:] if p.strip()]
    return []


def parse_release_page(html, logger):
    """
    Parse a BOM individual release page.

    BOM uses Amazon UI framework with consistent div structure:
    - "Opening" div contains: "$154,201,673" and "4,440 theaters"
    - "Domestic (" div contains: "38.4%", ")", "$652,980,194"
    - "Widest Release" div contains: "4,440 theaters"
    - "Distributor" div contains: "Walt Disney Studios Motion Pictures", "See full..."
    - "Genres" div contains genre names separated by whitespace

    Returns dict with all parsed fields.
    """
    soup = BeautifulSoup(html, "lxml")
    result = {}

    # --- Opening weekend gross + theaters ---
    opening_parts = get_summary_div_text(soup, "Opening")
    result["opening_wknd_gross"] = None
    result["opening_wknd_theaters"] = None
    for part in opening_parts:
        if "$" in part:
            result["opening_wknd_gross"] = parse_money(part)
        theater_match = re.search(r"([\d,]+)\s*theaters?", part, re.IGNORECASE)
        if theater_match:
            result["opening_wknd_theaters"] = parse_money(theater_match.group(1))

    # If theaters not found in parts, try full text near Opening
    if result["opening_wknd_theaters"] is None:
        page_text = soup.get_text()
        m = re.search(
            r"Opening.*?(\$[\d,]+).*?([\d,]+)\s*theaters",
            page_text, re.DOTALL | re.IGNORECASE
        )
        if m:
            if result["opening_wknd_gross"] is None:
                result["opening_wknd_gross"] = parse_money(m.group(1))
            result["opening_wknd_theaters"] = parse_money(m.group(2))

    # --- Widest release ---
    widest_parts = get_summary_div_text(soup, "Widest Release")
    result["widest_release"] = None
    for part in widest_parts:
        num_match = re.search(r"[\d,]+", part)
        if num_match:
            result["widest_release"] = parse_money(num_match.group())
            break

    # --- Domestic gross ---
    # The div text looks like: "Domestic (|38.4%|)|$652,980,194"
    domestic_parts = get_summary_div_text(soup, "Domestic (")
    result["domestic_gross"] = None
    for part in domestic_parts:
        if "$" in part:
            result["domestic_gross"] = parse_money(part)
            break

    # Fallback: regex on page text
    if result["domestic_gross"] is None:
        dom_match = re.search(
            r"Domestic\s*\([^)]*\)\s*\$?([\d,]+)",
            soup.get_text(), re.DOTALL
        )
        if dom_match:
            result["domestic_gross"] = parse_money(dom_match.group(1))

    # --- MPAA Rating ---
    mpaa_parts = get_summary_div_text(soup, "MPAA")
    if mpaa_parts:
        result["mpaa_rating"] = mpaa_parts[0]
    else:
        result["mpaa_rating"] = None

    # --- Genres ---
    genres_parts = get_summary_div_text(soup, "Genres")
    if not genres_parts:
        genres_parts = get_summary_div_text(soup, "Genre")
    if genres_parts:
        # Genres may come as separate parts or as one string with whitespace
        all_genres = []
        for part in genres_parts:
            # Split on whitespace in case they're concatenated
            for g in re.split(r"\s{2,}", part):
                g = g.strip()
                if g:
                    all_genres.append(g)
        result["genres"] = ", ".join(all_genres) if all_genres else None
    else:
        result["genres"] = None

    # --- Release Date ---
    date_parts = get_summary_div_text(soup, "Release Date")
    if date_parts:
        result["release_date"] = date_parts[0]
    else:
        # Look for date links with ?date= parameter
        date_link = soup.find("a", href=re.compile(r"\?date=\d{4}-\d{2}-\d{2}"))
        if date_link:
            result["release_date"] = date_link.get_text(strip=True)
        else:
            result["release_date"] = None

    # --- Distributor ---
    dist_parts = get_summary_div_text(soup, "Distributor")
    if dist_parts:
        # First part is the distributor name; exclude "See full company information"
        result["distributor"] = dist_parts[0]
    else:
        result["distributor"] = None

    return result


def scrape_bom_details():
    """Main function to scrape individual BOM release pages."""
    logger = setup_logging("02_scrape_bom_details")
    logger.info("Starting BOM details scrape")

    # Load the index
    index_path = config.DATA_RAW / "bom_index.csv"
    if not index_path.exists():
        logger.error(f"Index file not found: {index_path}. Run 01_scrape_bom_index.py first.")
        return

    index_df = pd.read_csv(index_path)
    logger.info(f"Loaded {len(index_df)} movies from index")

    # Check for existing checkpoint
    output_path = config.DATA_RAW / "bom_details.csv"
    _, processed_ids = load_checkpoint(output_path)
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} already scraped")

    # Scrape each movie's detail page
    batch = []
    total = len(index_df)
    skipped = 0

    for i, row in tqdm(index_df.iterrows(), total=total, desc="Scraping BOM details"):
        release_id = str(row["bom_release_id"])

        if release_id in processed_ids:
            skipped += 1
            continue

        url = row["release_url"]
        resp = make_request(url, delay=config.BOM_DELAY, logger=logger)

        if resp is None or resp.status_code != 200:
            logger.warning(f"Failed to fetch {url}")
            details = {
                "bom_release_id": release_id,
                "title": row["title"],
                "opening_wknd_gross": None,
                "opening_wknd_theaters": None,
                "widest_release": None,
                "domestic_gross": None,
                "mpaa_rating": None,
                "genres": None,
                "release_date": None,
                "distributor": None,
            }
        else:
            details = parse_release_page(resp.text, logger)
            details["bom_release_id"] = release_id
            details["title"] = row["title"]

        batch.append(details)

        # Checkpoint
        if len(batch) >= config.BOM_DETAILS_CHECKPOINT_INTERVAL:
            save_checkpoint(batch, output_path, logger)
            processed_ids.update(d["bom_release_id"] for d in batch)
            batch = []

    # Save remaining
    if batch:
        save_checkpoint(batch, output_path, logger)

    # Final summary
    if output_path.exists():
        final_df = pd.read_csv(output_path)
        n_with_opening = final_df["opening_wknd_gross"].notna().sum()
        logger.info(
            f"Done. {len(final_df)} total movies scraped. "
            f"{n_with_opening} have opening weekend data."
        )


if __name__ == "__main__":
    scrape_bom_details()
