"""
Step 3: Scrape The Numbers production budget database.

Paginates through the full budget table to build a comprehensive
lookup of movie budgets, domestic gross, and worldwide gross.

Output: data/raw/the_numbers_budgets.csv
"""

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import make_request, parse_money, parse_date, normalize_title, setup_logging


BASE_URL = "https://www.the-numbers.com/movie/budgets/all"
ROWS_PER_PAGE = 100


def parse_budget_page(html, logger):
    """Parse a single page of The Numbers budget table."""
    soup = BeautifulSoup(html, "lxml")
    movies = []

    # Find the main data table
    table = soup.find("table")
    if not table:
        logger.warning("No table found on page")
        return movies

    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 6:
            continue

        try:
            # Columns: Rank, Release Date, Movie, Production Budget,
            #          Domestic Gross, Worldwide Gross
            rank_text = cells[0].get_text(strip=True)
            release_date_text = cells[1].get_text(strip=True)
            title = cells[2].get_text(strip=True)
            budget_text = cells[3].get_text(strip=True)
            domestic_text = cells[4].get_text(strip=True)
            worldwide_text = cells[5].get_text(strip=True)

            # Parse rank
            rank = None
            if rank_text.isdigit():
                rank = int(rank_text)

            # Parse date and extract year
            release_date = parse_date(release_date_text)
            release_year = release_date.year if release_date else None

            movies.append({
                "tn_rank": rank,
                "title": title,
                "release_date": str(release_date) if release_date else release_date_text,
                "release_year": release_year,
                "production_budget": parse_money(budget_text),
                "domestic_gross": parse_money(domestic_text),
                "worldwide_gross": parse_money(worldwide_text),
                "title_normalized": normalize_title(title),
            })

        except Exception as e:
            logger.warning(f"Error parsing row: {e}")
            continue

    return movies


def scrape_the_numbers():
    """Main function to scrape The Numbers budget database."""
    logger = setup_logging("03_scrape_the_numbers")
    logger.info("Starting The Numbers budget scrape")

    all_movies = []
    offset = 1  # First page starts at 1

    # Paginate until we get an empty page
    with tqdm(desc="Scraping budget pages") as pbar:
        while True:
            if offset == 1:
                url = BASE_URL
            else:
                url = f"{BASE_URL}/{offset}"

            resp = make_request(url, delay=config.TN_DELAY, logger=logger)
            if resp is None or resp.status_code != 200:
                logger.warning(f"Failed to fetch page at offset {offset}")
                break

            movies = parse_budget_page(resp.text, logger)
            if not movies:
                logger.info(f"No more data at offset {offset}. Stopping.")
                break

            all_movies.extend(movies)
            pbar.update(1)
            pbar.set_postfix({"total_movies": len(all_movies)})

            # Move to next page
            if len(movies) < ROWS_PER_PAGE:
                logger.info(f"Last page had {len(movies)} rows (< {ROWS_PER_PAGE}). Done.")
                break
            offset += ROWS_PER_PAGE

    if not all_movies:
        logger.error("No movies scraped!")
        return

    df = pd.DataFrame(all_movies)

    # Save
    output_path = config.DATA_RAW / "the_numbers_budgets.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary stats
    logger.info(f"Saved {len(df)} movies to {output_path}")
    logger.info(f"Budget range: ${df['production_budget'].min():,.0f} - ${df['production_budget'].max():,.0f}")

    # Count movies in study window
    study_movies = df[
        (df["release_year"] >= 2021) & (df["release_year"] <= 2026)
    ]
    logger.info(f"Movies in study window (2021-2026): {len(study_movies)}")


if __name__ == "__main__":
    scrape_the_numbers()
