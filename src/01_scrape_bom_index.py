"""
Step 1: Scrape Box Office Mojo yearly index pages.

Fetches the yearly box office tables for 2021-2026 and extracts:
- Movie title, BOM release ID, total gross, max theaters, release date, distributor

Output: data/raw/bom_index.csv
"""

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import make_request, parse_money, setup_logging


BOM_YEAR_URL = "https://www.boxofficemojo.com/year/{year}/"


def parse_year_page(html, year, logger):
    """Parse a BOM yearly index page and return list of movie dicts."""
    soup = BeautifulSoup(html, "lxml")
    movies = []

    # Find the main data table
    table = soup.find("table")
    if not table:
        logger.error(f"No table found for year {year}")
        return movies

    rows = table.find_all("tr")
    # Skip header row
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        try:
            # Column 1: Release (title + link)
            title_cell = cells[1]
            title_link = title_cell.find("a")
            if not title_link:
                continue

            title = title_link.get_text(strip=True)
            href = title_link.get("href", "")

            # Extract release ID from href like /release/rl3638199041/
            id_match = re.search(r"/release/(rl\d+)/", href)
            if not id_match:
                logger.debug(f"No release ID in href: {href}")
                continue
            release_id = id_match.group(1)

            # Column 5: Gross (yearly gross)
            gross_text = cells[5].get_text(strip=True)
            gross = parse_money(gross_text)

            # Column 6: Theaters (max)
            theaters_text = cells[6].get_text(strip=True)
            theaters = parse_money(theaters_text)  # works for plain numbers too

            # Column 7: Total Gross
            total_gross_text = cells[7].get_text(strip=True)
            total_gross = parse_money(total_gross_text)

            # Column 8: Release Date (just month + day, need to combine with year)
            release_date_text = cells[8].get_text(strip=True)

            # Column 9: Distributor
            distributor = cells[9].get_text(strip=True)

            release_url = f"https://www.boxofficemojo.com/release/{release_id}/"

            movies.append({
                "bom_release_id": release_id,
                "title": title,
                "gross": gross,
                "total_gross": total_gross,
                "max_theaters": theaters,
                "release_date_raw": release_date_text,
                "bom_year": year,
                "distributor": distributor,
                "release_url": release_url,
            })

        except Exception as e:
            logger.warning(f"Error parsing row in year {year}: {e}")
            continue

    logger.info(f"Year {year}: parsed {len(movies)} movies")
    return movies


def scrape_bom_index():
    """Main function to scrape all BOM yearly index pages."""
    logger = setup_logging("01_scrape_bom_index")
    logger.info("Starting BOM index scrape")

    all_movies = []

    for year in config.BOM_YEARS:
        url = BOM_YEAR_URL.format(year=year)
        logger.info(f"Fetching {url}")

        resp = make_request(url, delay=config.BOM_DELAY, logger=logger)
        if resp is None:
            logger.error(f"Failed to fetch year {year}")
            continue

        movies = parse_year_page(resp.text, year, logger)
        all_movies.extend(movies)

    if not all_movies:
        logger.error("No movies scraped!")
        return

    df = pd.DataFrame(all_movies)

    # Deduplicate on bom_release_id (Dec releases may appear on two year pages)
    before = len(df)
    df = df.drop_duplicates(subset="bom_release_id", keep="first")
    after = len(df)
    if before != after:
        logger.info(f"Removed {before - after} duplicate release IDs")

    # Save
    output_path = config.DATA_RAW / "bom_index.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} movies to {output_path}")

    # Summary
    for year in config.BOM_YEARS:
        count = len(df[df["bom_year"] == year])
        logger.info(f"  Year {year}: {count} movies")


if __name__ == "__main__":
    scrape_bom_index()
