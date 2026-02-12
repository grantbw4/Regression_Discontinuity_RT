"""
Step 5: Merge all scraped datasets into a single analysis-ready CSV.

Joins BOM index + details (exact), BOM + RT scores (exact), and
BOM + The Numbers budgets (fuzzy title match). Applies study filters
and constructs RDD variables.

Input:  data/raw/bom_index.csv, bom_details.csv, rt_scores.csv, the_numbers_budgets.csv
Output: data/processed/merged_dataset.csv, data/processed/match_diagnostics.csv
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import normalize_title, parse_date, setup_logging


def load_raw_data(logger):
    """Load all four raw CSV files."""
    bom_index = pd.read_csv(config.DATA_RAW / "bom_index.csv")
    logger.info(f"BOM index: {len(bom_index)} rows")

    bom_details = pd.read_csv(config.DATA_RAW / "bom_details.csv")
    logger.info(f"BOM details: {len(bom_details)} rows")

    rt_scores = pd.read_csv(config.DATA_RAW / "rt_scores.csv")
    logger.info(f"RT scores: {len(rt_scores)} rows")

    tn_budgets = pd.read_csv(config.DATA_RAW / "the_numbers_budgets.csv")
    logger.info(f"The Numbers budgets: {len(tn_budgets)} rows")

    return bom_index, bom_details, rt_scores, tn_budgets


def merge_bom(bom_index, bom_details, logger):
    """Merge BOM index and details on bom_release_id."""
    # Use details as the primary source; supplement with index data
    bom = pd.merge(
        bom_index[["bom_release_id", "title", "total_gross", "max_theaters",
                    "release_date_raw", "bom_year", "distributor", "release_url"]],
        bom_details[["bom_release_id", "opening_wknd_gross", "opening_wknd_theaters",
                      "widest_release", "domestic_gross", "mpaa_rating", "genres",
                      "release_date", "distributor"]],
        on="bom_release_id",
        how="left",
        suffixes=("_index", "_detail"),
    )

    # Prefer detail-level data, fall back to index
    bom["title"] = bom["title"]
    bom["domestic_gross"] = bom["domestic_gross"].fillna(bom["total_gross"])
    bom["distributor"] = bom["distributor_detail"].fillna(bom["distributor_index"])

    # Parse release date — prefer detailed, fall back to index raw
    def resolve_release_date(row):
        # Try detailed release date first
        if pd.notna(row.get("release_date")):
            d = parse_date(str(row["release_date"]))
            if d:
                return d
        # Fall back to index raw date + year
        if pd.notna(row.get("release_date_raw")) and pd.notna(row.get("bom_year")):
            raw = str(row["release_date_raw"]).strip()
            year = int(row["bom_year"])
            d = parse_date(f"{raw}, {year}")
            if d:
                return d
        return None

    bom["release_date_parsed"] = bom.apply(resolve_release_date, axis=1)

    # Clean up columns
    bom = bom.rename(columns={"mpaa_rating": "mpaa_rating"})
    keep_cols = [
        "bom_release_id", "title", "release_date_parsed", "bom_year",
        "opening_wknd_gross", "opening_wknd_theaters", "domestic_gross",
        "widest_release", "max_theaters", "distributor", "mpaa_rating", "genres",
    ]
    bom = bom[[c for c in keep_cols if c in bom.columns]]

    logger.info(f"Merged BOM: {len(bom)} rows")
    return bom


def merge_rt(bom, rt_scores, logger):
    """Merge RT scores onto BOM data via bom_release_id."""
    rt_cols = [
        "bom_release_id", "tomatometer", "audience_score",
        "critic_count", "audience_count", "rt_genres", "rt_rating",
        "match_method", "rt_url", "rt_title",
    ]
    rt_subset = rt_scores[[c for c in rt_cols if c in rt_scores.columns]]

    merged = pd.merge(bom, rt_subset, on="bom_release_id", how="left")

    # Fill MPAA rating from RT if missing from BOM
    if "rt_rating" in merged.columns:
        merged["mpaa_rating"] = merged["mpaa_rating"].fillna(merged["rt_rating"])

    n_with_tm = merged["tomatometer"].notna().sum()
    n_with_as = merged["audience_score"].notna().sum()
    logger.info(f"After RT merge: {n_with_tm} with Tomatometer, {n_with_as} with Audience Score")

    return merged


def fuzzy_match_budgets(bom_df, tn_df, logger):
    """
    Fuzzy-match The Numbers budget data to BOM movies.

    Uses rapidfuzz token_sort_ratio with year blocking for precision.
    """
    # Pre-normalize BOM titles
    bom_df = bom_df.copy()
    bom_df["title_norm"] = bom_df["title"].apply(normalize_title)

    # Extract release year from parsed date
    bom_df["release_year"] = bom_df["release_date_parsed"].apply(
        lambda d: d.year if d else None
    )

    # Ensure TN has normalized titles and years
    tn_df = tn_df.copy()
    if "title_normalized" not in tn_df.columns:
        tn_df["title_normalized"] = tn_df["title"].apply(normalize_title)
    if "release_year" not in tn_df.columns:
        tn_df["release_year"] = pd.to_numeric(tn_df["release_year"], errors="coerce")

    # Build match results
    match_results = []

    for idx, bom_row in bom_df.iterrows():
        bom_title = bom_row["title_norm"]
        bom_year = bom_row["release_year"]

        if not bom_title:
            match_results.append({
                "idx": idx,
                "tn_title_matched": None,
                "tn_match_score": 0,
                "production_budget": None,
                "tn_domestic_gross": None,
                "tn_match_status": "unmatched",
            })
            continue

        # Year blocking: consider only TN movies within ±1 year
        if bom_year and pd.notna(bom_year):
            year_mask = tn_df["release_year"].between(bom_year - 1, bom_year + 1)
            candidates = tn_df[year_mask]
        else:
            candidates = tn_df

        if candidates.empty:
            match_results.append({
                "idx": idx,
                "tn_title_matched": None,
                "tn_match_score": 0,
                "production_budget": None,
                "tn_domestic_gross": None,
                "tn_match_status": "unmatched",
            })
            continue

        # Find best match using token_sort_ratio
        result = process.extractOne(
            bom_title,
            candidates["title_normalized"].tolist(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=0,
        )

        if result is None:
            match_results.append({
                "idx": idx,
                "tn_title_matched": None,
                "tn_match_score": 0,
                "production_budget": None,
                "tn_domestic_gross": None,
                "tn_match_status": "unmatched",
            })
            continue

        matched_title_norm, score, match_idx = result
        tn_row = candidates.iloc[match_idx]

        if score >= config.FUZZY_MATCH_ACCEPT_THRESHOLD:
            status = "matched"
        elif score >= config.FUZZY_MATCH_REVIEW_THRESHOLD:
            status = "review"
        else:
            status = "unmatched"

        match_results.append({
            "idx": idx,
            "tn_title_matched": tn_row["title"],
            "tn_match_score": score,
            "production_budget": tn_row.get("production_budget"),
            "tn_domestic_gross": tn_row.get("domestic_gross"),
            "tn_match_status": status,
        })

    match_df = pd.DataFrame(match_results).set_index("idx")

    # Join match results back to BOM data
    for col in ["tn_title_matched", "tn_match_score", "production_budget",
                "tn_domestic_gross", "tn_match_status"]:
        bom_df[col] = match_df[col]

    n_matched = (bom_df["tn_match_status"] == "matched").sum()
    n_review = (bom_df["tn_match_status"] == "review").sum()
    n_unmatched = (bom_df["tn_match_status"] == "unmatched").sum()
    logger.info(
        f"Budget matching: {n_matched} matched, {n_review} for review, "
        f"{n_unmatched} unmatched"
    )

    return bom_df


def apply_study_filters(df, logger):
    """Filter to study window and wide releases."""
    before = len(df)

    # Date filter
    start = date.fromisoformat(config.START_DATE)
    end = date.fromisoformat(config.END_DATE)
    df = df[df["release_date_parsed"].apply(
        lambda d: d is not None and start <= d <= end
    )]
    logger.info(f"After date filter ({config.START_DATE} to {config.END_DATE}): {len(df)} rows (removed {before - len(df)})")

    # Wide release filter
    before2 = len(df)
    df = df[df["opening_wknd_theaters"] >= config.MIN_OPENING_THEATERS]
    logger.info(f"After wide release filter (>={config.MIN_OPENING_THEATERS} theaters): {len(df)} rows (removed {before2 - len(df)})")

    return df


def construct_rdd_variables(df, logger):
    """Create all derived variables needed for the RD analysis."""
    # RDD running variables (centered at threshold)
    df["tomatometer_centered"] = df["tomatometer"] - config.RD_THRESHOLD
    df["audience_score_centered"] = df["audience_score"] - config.RD_THRESHOLD

    # Treatment indicators
    df["is_fresh_critic"] = (df["tomatometer"] >= config.RD_THRESHOLD).astype("Int64")
    df["is_fresh_audience"] = (df["audience_score"] >= config.RD_THRESHOLD).astype("Int64")

    # Log-transformed outcome variables
    df["log_opening_gross"] = np.log(df["opening_wknd_gross"].clip(lower=1).astype(float))
    df["log_total_gross"] = np.log(df["domestic_gross"].clip(lower=1).astype(float))

    # Log-transformed control variables
    df["log_theaters"] = np.log(df["opening_wknd_theaters"].clip(lower=1).astype(float))
    df["log_budget"] = np.where(
        df["production_budget"].notna() & (df["production_budget"] > 0),
        np.log(df["production_budget"].clip(lower=1).astype(float)),
        np.nan,
    )

    # Calendar variables
    df["release_year"] = df["release_date_parsed"].apply(lambda d: d.year if d else None)
    df["release_month"] = df["release_date_parsed"].apply(lambda d: d.month if d else None)

    # In-theaters flag
    cutoff = date.fromisoformat(config.END_DATE) - timedelta(days=config.IN_THEATERS_WINDOW_DAYS)
    df["in_theaters"] = df["release_date_parsed"].apply(
        lambda d: d is not None and d >= cutoff
    )

    logger.info(f"Constructed RDD variables. {df['is_fresh_critic'].notna().sum()} have critic treatment, "
                f"{df['is_fresh_audience'].notna().sum()} have audience treatment.")

    return df


def create_match_diagnostics(df, logger):
    """Create a diagnostics CSV for manual review of matches."""
    diag_cols = [
        "bom_release_id", "title", "release_year",
        "tomatometer", "audience_score", "match_method", "rt_url", "rt_title",
        "tn_title_matched", "tn_match_score", "tn_match_status", "production_budget",
    ]
    diag = df[[c for c in diag_cols if c in df.columns]].copy()

    diag_path = config.DATA_PROCESSED / "match_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    logger.info(f"Saved match diagnostics to {diag_path}")

    # Log movies needing review
    needs_review = diag[
        (diag.get("match_method") == "unmatched") |
        (diag.get("tn_match_status") == "review")
    ]
    if len(needs_review) > 0:
        logger.info(f"{len(needs_review)} movies need manual review — see match_diagnostics.csv")


def merge_datasets():
    """Main merge pipeline."""
    logger = setup_logging("05_merge_datasets")
    logger.info("Starting dataset merge")

    # Load raw data
    bom_index, bom_details, rt_scores, tn_budgets = load_raw_data(logger)

    # Step A: Merge BOM index + details
    bom = merge_bom(bom_index, bom_details, logger)

    # Step B: Merge RT scores
    bom_rt = merge_rt(bom, rt_scores, logger)

    # Step C: Fuzzy match budgets
    full = fuzzy_match_budgets(bom_rt, tn_budgets, logger)

    # Step D: Apply study filters
    filtered = apply_study_filters(full, logger)

    # Step E: Construct RDD variables
    final = construct_rdd_variables(filtered, logger)

    # Convert release_date_parsed to string for CSV
    final["release_date"] = final["release_date_parsed"].apply(
        lambda d: str(d) if d else None
    )

    # Select final columns
    output_cols = [
        "bom_release_id", "title", "release_date", "release_year", "release_month",
        "distributor", "mpaa_rating", "genres", "rt_genres",
        "opening_wknd_gross", "opening_wknd_theaters", "domestic_gross",
        "widest_release", "max_theaters", "production_budget",
        "tomatometer", "audience_score", "critic_count", "audience_count",
        "is_fresh_critic", "is_fresh_audience",
        "tomatometer_centered", "audience_score_centered",
        "log_opening_gross", "log_total_gross", "log_theaters", "log_budget",
        "in_theaters",
        "match_method", "tn_match_score", "tn_match_status",
    ]
    final_out = final[[c for c in output_cols if c in final.columns]]

    # Save
    output_path = config.DATA_PROCESSED / "merged_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_out.to_csv(output_path, index=False)
    logger.info(f"Saved {len(final_out)} movies to {output_path}")

    # Create diagnostics
    create_match_diagnostics(final, logger)

    # Final summary
    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"Total movies in analysis dataset: {len(final_out)}")
    logger.info(f"With Tomatometer: {final_out['tomatometer'].notna().sum()}")
    logger.info(f"With Audience Score: {final_out['audience_score'].notna().sum()}")
    logger.info(f"With Budget: {final_out['production_budget'].notna().sum()}")
    logger.info(f"In-theaters (incomplete gross): {final_out['in_theaters'].sum()}")
    if "is_fresh_critic" in final_out.columns:
        logger.info(f"Fresh (critic): {(final_out['is_fresh_critic'] == 1).sum()}")
        logger.info(f"Rotten (critic): {(final_out['is_fresh_critic'] == 0).sum()}")
    if "is_fresh_audience" in final_out.columns:
        logger.info(f"Fresh (audience): {(final_out['is_fresh_audience'] == 1).sum()}")
        logger.info(f"Rotten (audience): {(final_out['is_fresh_audience'] == 0).sum()}")


if __name__ == "__main__":
    merge_datasets()
