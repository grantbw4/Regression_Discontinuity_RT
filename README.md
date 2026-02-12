# Does anyone else care if the movie's "Fresh?"

A regression discontinuity analysis of whether Rotten Tomatoes' "Fresh" vs. "Rotten" label has a causal effect on box office revenue in the post-COVID era (Sep 2021 - Feb 2026).

**[View the full report](https://grantbw4.github.io/Regression_Discontinuity_RT/)**

## Motivation

Rotten Tomatoes labels films "Fresh" (score >= 60%) or "Rotten" (score < 60%). This sharp cutoff creates a natural experiment: if the binary label itself drives ticket sales, we'd expect a discontinuous jump in revenue right at 60%, beyond what the continuous score predicts.

Nishijima, Rodrigues & Souza ([2021](https://www.tandfonline.com/doi/full/10.1080/13504851.2021.1918324)) ran this analysis on the Tomatometer using 1,239 films from 1999-2019 and found no effect. I wanted to revisit this in the post-COVID era and extend it to the Audience Score, which their study didn't cover.

## Results

No discontinuity at either threshold. The "Fresh" label doesn't appear to cause a detectable bump in box office revenue for either critic or audience scores. See the [full report](https://grantbw4.github.io/Regression_Discontinuity_RT/) for tables, plots, and discussion.

## Data Sources

| Source | What | How |
|--------|------|-----|
| [Box Office Mojo](https://www.boxofficemojo.com) | Film list, opening weekend/total domestic gross, theater counts, release dates | Scraped yearly index + individual release pages |
| [Rotten Tomatoes](https://www.rottentomatoes.com) | Tomatometer, Audience Score, review counts | Matched via URL slug construction with search fallback |
| [The Numbers](https://www.the-numbers.com) | Production budgets | Fuzzy-matched by title and release year |

Final dataset: **621 wide-release films** (600+ opening theaters).

## Pipeline

Scripts are numbered and meant to be run in order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/01_scrape_bom_index.py` | Scrape Box Office Mojo yearly index pages |
| 2 | `src/02_scrape_bom_details.py` | Scrape individual BOM release pages for detailed stats |
| 3 | `src/03_scrape_the_numbers.py` | Scrape production budgets from The Numbers |
| 4 | `src/04_scrape_rotten_tomatoes.py` | Scrape RT scores for each film |
| 5 | `src/05_merge_datasets.py` | Merge all sources, fuzzy-match budgets, construct RDD variables |
| 6 | `src/06_rdd_analysis.py` | Run RDD regressions (rdrobust + OLS robustness checks) |
| 7 | `src/07_rdd_report.py` | Generate self-contained HTML report |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install statsmodels rdrobust matplotlib
```

## References

- Calonico, S., Cattaneo, M.D. & Titiunik, R. (2014). "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs." *Econometrica*, 82(6), 2295-2326.
- Nishijima, M., Rodrigues, M. & Souza, T.L.D. (2021). "Is Rotten Tomatoes killing the movie industry? A regression discontinuity approach." *Applied Economics Letters*, 29(13), 1187-1192.
