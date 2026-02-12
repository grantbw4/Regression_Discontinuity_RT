"""
Step 6: RDD Analysis — Estimate the causal effect of Fresh certification on box office.

Runs 24 regressions: 2 running variables (critic, audience) × 2 outcomes
(log_opening_gross, log_total_gross) × 2 methods (rdrobust, OLS polynomial)
× 2 control sets (none, full). OLS includes both linear and quadratic.

For log_total_gross, movies still in theaters (in_theaters == True) are excluded
since their domestic gross is incomplete.

Output: formatted results tables printed to console and saved to output/rdd_results.txt
"""

import sys
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from rdrobust import rdrobust

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """Load merged dataset and prepare samples."""
    df = pd.read_csv(config.DATA_PROCESSED / "merged_dataset.csv")

    # MPAA dummies (drop G as reference — fewest obs)
    mpaa_dummies = pd.get_dummies(df["mpaa_rating"], prefix="mpaa", drop_first=False)
    mpaa_dummies = mpaa_dummies.drop(columns=["mpaa_G"], errors="ignore")

    # Year dummies (drop earliest year as reference)
    year_dummies = pd.get_dummies(df["release_year"], prefix="year", drop_first=True)

    df = pd.concat([df, mpaa_dummies, year_dummies], axis=1)

    return df


def get_control_cols(df):
    """Return list of control column names present in df."""
    controls = ["log_budget", "log_theaters"]
    # MPAA dummies (exclude the original string column)
    controls += [c for c in df.columns if c.startswith("mpaa_") and c != "mpaa_rating"]
    controls += [c for c in df.columns if c.startswith("year_")]
    return [c for c in controls if c in df.columns]


# ── rdrobust wrapper ──────────────────────────────────────────────────────────

def run_rdrobust(y, x, covs=None):
    """Run rdrobust and return a results dict."""
    y_arr = y.values.astype(float)
    x_arr = x.values.astype(float)

    kwargs = dict(c=0, kernel="tri", bwselect="mserd")

    if covs is not None:
        covs_arr = covs.values.astype(float)
        kwargs["covs"] = covs_arr

    try:
        result = rdrobust(y_arr, x_arr, **kwargs)
        out = {
            "coef_conv": float(result.coef.iloc[0, 0]),
            "coef_bc": float(result.coef.iloc[1, 0]),
            "coef_robust": float(result.coef.iloc[2, 0]),
            "se_conv": float(result.se.iloc[0, 0]),
            "se_robust": float(result.se.iloc[2, 0]),
            "pv_conv": float(result.pv.iloc[0, 0]),
            "pv_robust": float(result.pv.iloc[2, 0]),
            "ci_lower": float(result.ci.iloc[2, 0]),
            "ci_upper": float(result.ci.iloc[2, 1]),
            "bw_h": float(result.bws.iloc[0, 0]),
            "N_left": int(result.N_h[0]),
            "N_right": int(result.N_h[1]),
            "N_total": int(result.N_h[0]) + int(result.N_h[1]),
        }
        return out
    except Exception as e:
        import traceback
        return {"error": f"{e}\n{traceback.format_exc()}"}


# ── Parametric OLS RDD ───────────────────────────────────────────────────────

def run_ols_rdd(y, x, treatment, order=1, covs=None):
    """
    Run parametric RDD via OLS.

    Y = α + τ D + β₁ X + β₂ (D×X) [+ β₃ X² + β₄ (D×X²)] [+ Z γ] + ε

    Returns dict with τ estimate, SE, p-value, CI.
    """
    data = pd.DataFrame({"y": y.values, "x": x.values, "D": treatment.values})

    # Polynomial terms interacted with treatment
    regressors = ["D", "x", "Dx"]
    data["Dx"] = data["D"] * data["x"]

    if order >= 2:
        data["x2"] = data["x"] ** 2
        data["Dx2"] = data["D"] * data["x2"]
        regressors += ["x2", "Dx2"]

    if covs is not None:
        covs_reset = covs.reset_index(drop=True)
        for col in covs_reset.columns:
            data[col] = pd.to_numeric(covs_reset[col], errors="coerce")
            regressors.append(col)

    data = data.dropna()

    X = sm.add_constant(data[regressors].astype(float))
    model = sm.OLS(data["y"], X).fit(cov_type="HC1")

    tau = model.params["D"]
    se = model.bse["D"]
    pv = model.pvalues["D"]
    ci = model.conf_int().loc["D"]

    return {
        "coef": tau,
        "se": se,
        "pv": pv,
        "ci_lower": ci.iloc[0],
        "ci_upper": ci.iloc[1],
        "N": int(model.nobs),
        "r2": model.rsquared,
    }


# ── Run all specifications ────────────────────────────────────────────────────

def run_all(df):
    """Run all 24 regressions and collect results."""
    control_cols = get_control_cols(df)
    results = []

    specs = [
        ("Critic",   "tomatometer_centered", "is_fresh_critic",   "tomatometer"),
        ("Audience", "audience_score_centered", "is_fresh_audience", "audience_score"),
    ]

    outcomes = [
        ("log_opening_gross", "Log Opening Gross", None),
        ("log_total_gross",   "Log Total Gross",   True),  # exclude in_theaters
    ]

    for score_label, running_var, treatment_var, score_col in specs:
        for outcome_var, outcome_label, exclude_in_theaters in outcomes:

            # Build sample
            sample = df.dropna(subset=[running_var, treatment_var, outcome_var])
            if exclude_in_theaters:
                sample = sample[sample["in_theaters"] == False]

            y = sample[outcome_var]
            x = sample[running_var]
            d = sample[treatment_var]
            covs_df = sample[control_cols].dropna(axis=1)
            # Align covs with sample (drop rows with NaN in any control)
            valid_mask = covs_df.notna().all(axis=1)
            sample_ctrl = sample[valid_mask]
            y_ctrl = sample_ctrl[outcome_var]
            x_ctrl = sample_ctrl[running_var]
            d_ctrl = sample_ctrl[treatment_var]
            covs_clean = covs_df[valid_mask]

            n_sample = len(sample)
            n_ctrl = len(sample_ctrl)

            # ── rdrobust, no controls ──
            rd = run_rdrobust(y, x)
            results.append({
                "Score": score_label,
                "Outcome": outcome_label,
                "Method": "rdrobust",
                "Controls": "No",
                "Coef": rd.get("coef_robust") or rd.get("coef_conv"),
                "SE": rd.get("se_robust") or rd.get("se_conv"),
                "p-value": rd.get("pv_robust") or rd.get("pv_conv"),
                "CI Lower": rd.get("ci_lower"),
                "CI Upper": rd.get("ci_upper"),
                "BW": rd.get("bw_h"),
                "N (eff)": rd.get("N_total"),
                "N": n_sample,
                "Error": rd.get("error"),
            })

            # ── rdrobust, with controls ──
            rd_c = run_rdrobust(y_ctrl, x_ctrl, covs=covs_clean)
            results.append({
                "Score": score_label,
                "Outcome": outcome_label,
                "Method": "rdrobust",
                "Controls": "Yes",
                "Coef": rd_c.get("coef_robust") or rd_c.get("coef_conv"),
                "SE": rd_c.get("se_robust") or rd_c.get("se_conv"),
                "p-value": rd_c.get("pv_robust") or rd_c.get("pv_conv"),
                "CI Lower": rd_c.get("ci_lower"),
                "CI Upper": rd_c.get("ci_upper"),
                "BW": rd_c.get("bw_h"),
                "N (eff)": rd_c.get("N_total"),
                "N": n_ctrl,
                "Error": rd_c.get("error"),
            })

            # ── OLS linear, no controls ──
            ols_l = run_ols_rdd(y, x, d, order=1)
            results.append({
                "Score": score_label,
                "Outcome": outcome_label,
                "Method": "OLS Linear",
                "Controls": "No",
                "Coef": ols_l["coef"],
                "SE": ols_l["se"],
                "p-value": ols_l["pv"],
                "CI Lower": ols_l["ci_lower"],
                "CI Upper": ols_l["ci_upper"],
                "N": ols_l["N"],
                "R²": ols_l["r2"],
            })

            # ── OLS linear, with controls ──
            ols_lc = run_ols_rdd(y_ctrl, x_ctrl, d_ctrl, order=1, covs=covs_clean)
            results.append({
                "Score": score_label,
                "Outcome": outcome_label,
                "Method": "OLS Linear",
                "Controls": "Yes",
                "Coef": ols_lc["coef"],
                "SE": ols_lc["se"],
                "p-value": ols_lc["pv"],
                "CI Lower": ols_lc["ci_lower"],
                "CI Upper": ols_lc["ci_upper"],
                "N": ols_lc["N"],
                "R²": ols_lc["r2"],
            })

            # ── OLS quadratic, no controls ──
            ols_q = run_ols_rdd(y, x, d, order=2)
            results.append({
                "Score": score_label,
                "Outcome": outcome_label,
                "Method": "OLS Quadratic",
                "Controls": "No",
                "Coef": ols_q["coef"],
                "SE": ols_q["se"],
                "p-value": ols_q["pv"],
                "CI Lower": ols_q["ci_lower"],
                "CI Upper": ols_q["ci_upper"],
                "N": ols_q["N"],
                "R²": ols_q["r2"],
            })

            # ── OLS quadratic, with controls ──
            ols_qc = run_ols_rdd(y_ctrl, x_ctrl, d_ctrl, order=2, covs=covs_clean)
            results.append({
                "Score": score_label,
                "Outcome": outcome_label,
                "Method": "OLS Quadratic",
                "Controls": "Yes",
                "Coef": ols_qc["coef"],
                "SE": ols_qc["se"],
                "p-value": ols_qc["pv"],
                "CI Lower": ols_qc["ci_lower"],
                "CI Upper": ols_qc["ci_upper"],
                "N": ols_qc["N"],
                "R²": ols_qc["r2"],
            })

    return pd.DataFrame(results)


# ── Formatting & output ──────────────────────────────────────────────────────

def stars(pv):
    """Significance stars."""
    if pv is None or pd.isna(pv):
        return ""
    if pv < 0.01:
        return "***"
    if pv < 0.05:
        return "**"
    if pv < 0.10:
        return "*"
    return ""


def format_results_table(results_df):
    """Format results into readable tables, grouped by score type and outcome."""
    buf = StringIO()

    buf.write("=" * 90 + "\n")
    buf.write("RDD ESTIMATION RESULTS\n")
    buf.write(f"Rotten Tomatoes Fresh Certification → Box Office Revenue\n")
    buf.write(f"Cutoff: {config.RD_THRESHOLD}%  |  Study Period: {config.START_DATE} to {config.END_DATE}\n")
    buf.write(f"Wide releases only (≥{config.MIN_OPENING_THEATERS} opening theaters)\n")
    buf.write("=" * 90 + "\n\n")

    for score_label in ["Critic", "Audience"]:
        score_var = "Tomatometer" if score_label == "Critic" else "Audience Score"
        buf.write("=" * 90 + "\n")
        buf.write(f"PANEL: {score_label.upper()} SCORE ({score_var} centered at {config.RD_THRESHOLD})\n")
        buf.write("=" * 90 + "\n\n")

        for outcome_label in ["Log Opening Gross", "Log Total Gross"]:
            subset = results_df[
                (results_df["Score"] == score_label) &
                (results_df["Outcome"] == outcome_label)
            ]
            if subset.empty:
                continue

            sample_note = ""
            if outcome_label == "Log Total Gross":
                sample_note = " (excluding movies still in theaters)"

            buf.write(f"  Outcome: {outcome_label}{sample_note}\n")
            buf.write("  " + "-" * 86 + "\n")
            buf.write(f"  {'Method':<20} {'Controls':<10} {'Coef':>10} {'SE':>10} "
                      f"{'p-val':>8} {'':>4} {'95% CI':>22} {'N':>8} {'BW':>7}\n")
            buf.write("  " + "-" * 86 + "\n")

            for _, row in subset.iterrows():
                if pd.notna(row.get("Error")):
                    buf.write(f"  {row['Method']:<20} {row['Controls']:<10} "
                              f"ERROR: {row['Error']}\n")
                    continue

                coef_str = f"{row['Coef']:>10.4f}" if pd.notna(row.get("Coef")) else f"{'—':>10}"
                se_str = f"{row['SE']:>10.4f}" if pd.notna(row.get("SE")) else f"{'—':>10}"
                pv_str = f"{row['p-value']:>8.4f}" if pd.notna(row.get("p-value")) else f"{'—':>8}"
                star_str = f"{stars(row.get('p-value')):<4}"

                ci_l = row.get("CI Lower")
                ci_u = row.get("CI Upper")
                if pd.notna(ci_l) and pd.notna(ci_u):
                    ci_str = f"[{ci_l:>8.4f}, {ci_u:>8.4f}]"
                else:
                    ci_str = f"{'—':>22}"

                n_str = ""
                if pd.notna(row.get("N (eff)")):
                    n_str = f"{int(row['N (eff)']):>8}"
                elif pd.notna(row.get("N")):
                    n_str = f"{int(row['N']):>8}"

                bw_str = f"{row['BW']:>7.2f}" if pd.notna(row.get("BW")) else f"{'':>7}"

                buf.write(f"  {row['Method']:<20} {row['Controls']:<10} "
                          f"{coef_str} {se_str} {pv_str} {star_str}{ci_str} {n_str} {bw_str}\n")

            buf.write("  " + "-" * 86 + "\n\n")

        buf.write("\n")

    buf.write("=" * 90 + "\n")
    buf.write("Notes:\n")
    buf.write("  * p < 0.10, ** p < 0.05, *** p < 0.01\n")
    buf.write("  rdrobust: Robust bias-corrected estimates (Calonico, Cattaneo, Titiunik 2014).\n")
    buf.write("    N = effective sample within MSE-optimal bandwidth. BW = bandwidth.\n")
    buf.write("  OLS: HC1 robust standard errors. Full score range.\n")
    buf.write("  Controls: log(budget), log(theaters), MPAA rating dummies, year dummies.\n")
    buf.write("  Coefficients are in log points (multiply by 100 for approx. % effect).\n")
    buf.write("=" * 90 + "\n")

    return buf.getvalue()


def main():
    df = load_data()

    print("Running 24 RDD specifications...\n")
    results_df = run_all(df)

    output_text = format_results_table(results_df)
    print(output_text)

    # Save results
    output_dir = Path(config.PROJECT_ROOT / "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "rdd_results.txt", "w") as f:
        f.write(output_text)

    results_df.to_csv(output_dir / "rdd_results_raw.csv", index=False)
    print(f"\nSaved to {output_dir / 'rdd_results.txt'} and {output_dir / 'rdd_results_raw.csv'}")


if __name__ == "__main__":
    main()
