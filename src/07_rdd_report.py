"""
Step 7: Generate a self-contained HTML report of RDD results.

Reads the merged dataset and raw regression results, generates diagnostic
plots (score density, RDD scatters), and produces a single HTML file with
everything embedded (base64 images, inline CSS).

Output: output/rdd_report.html
"""

import base64
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Plot helpers ──────────────────────────────────────────────────────────────

def fig_to_base64(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_density(df, col, label, threshold=60):
    """Histogram of scores around the cutoff."""
    scores = df[col].dropna()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    bins = np.arange(scores.min() - 0.5, scores.max() + 1.5, 1)
    colors = np.where(bins[:-1] >= threshold, "#4a90d9", "#d94a4a")

    n, _, patches = ax.hist(scores, bins=bins, edgecolor="white", linewidth=0.5)
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    ax.axvline(threshold, color="#333", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(threshold + 0.5, ax.get_ylim()[1] * 0.92, f"Cutoff = {threshold}%",
            fontsize=9, color="#333", ha="left")

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Number of Films", fontsize=10)
    ax.set_title(f"Distribution of {label}", fontsize=11, fontweight="600")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_rdd_scatter(df, running_col, outcome_col, running_label, outcome_label, threshold=60):
    """Binned scatter with local polynomial fits on each side."""
    data = df[[running_col, outcome_col]].dropna()
    x = data[running_col].values  # already centered
    y = data[outcome_col].values

    fig, ax = plt.subplots(figsize=(6, 4))

    # Bin scatter (2-unit bins)
    bin_width = 2
    x_min, x_max = x.min(), x.max()
    bin_edges = np.arange(x_min - 0.5, x_max + bin_width + 0.5, bin_width)
    bin_centers = []
    bin_means = []
    bin_colors = []
    for i in range(len(bin_edges) - 1):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if mask.sum() >= 3:
            bc = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_centers.append(bc)
            bin_means.append(y[mask].mean())
            bin_colors.append("#4a90d9" if bc >= 0 else "#d94a4a")

    ax.scatter(bin_centers, bin_means, c=bin_colors, s=30, zorder=5, edgecolors="white", linewidth=0.5)

    # Polynomial fits (degree 2) on each side
    for side, color in [("left", "#d94a4a"), ("right", "#4a90d9")]:
        mask = x < 0 if side == "left" else x >= 0
        xs, ys = x[mask], y[mask]
        if len(xs) < 5:
            continue
        coeffs = np.polyfit(xs, ys, 2)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(x_fit, poly(x_fit), color=color, linewidth=2, alpha=0.85)

    ax.axvline(0, color="#333", linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_xlabel(f"{running_label} (centered at {threshold}%)", fontsize=10)
    ax.set_ylabel(outcome_label, fontsize=10)
    ax.set_title(f"{outcome_label} vs. {running_label}", fontsize=11, fontweight="600")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig_to_base64(fig)


# ── Table formatting ──────────────────────────────────────────────────────────

def stars(pv):
    if pd.isna(pv):
        return ""
    if pv < 0.01:
        return "***"
    if pv < 0.05:
        return "**"
    if pv < 0.10:
        return "*"
    return ""


def stars_html(pv):
    s = stars(pv)
    if not s:
        return ""
    return f'<span class="stars">{s}</span>'


def fmt(val, decimals=4, is_int=False):
    if pd.isna(val):
        return "&mdash;"
    if is_int:
        return f"{int(val):,}"
    return f"{val:.{decimals}f}"


def build_results_table(results_df, score, outcome):
    subset = results_df[
        (results_df["Score"] == score) & (results_df["Outcome"] == outcome)
    ]
    if subset.empty:
        return ""

    rows_html = ""
    for _, r in subset.iterrows():
        if pd.notna(r.get("Error")):
            continue
        method = r["Method"]
        ctrl = r["Controls"]
        coef = fmt(r["Coef"])
        se = fmt(r["SE"])
        pv = fmt(r["p-value"])
        st = stars_html(r["p-value"])
        ci_l = fmt(r.get("CI Lower"))
        ci_u = fmt(r.get("CI Upper"))
        ci = f"[{ci_l}, {ci_u}]"

        n_val = r.get("N (eff)") if pd.notna(r.get("N (eff)")) else r.get("N")
        n = fmt(n_val, is_int=True) if pd.notna(n_val) else "&mdash;"
        bw = fmt(r.get("BW"), decimals=2) if pd.notna(r.get("BW")) else "&mdash;"

        # Highlight preferred spec
        is_preferred = method == "rdrobust" and ctrl == "Yes"
        row_class = ' class="preferred"' if is_preferred else ""

        rows_html += f"""        <tr{row_class}>
          <td>{method}</td>
          <td>{ctrl}</td>
          <td class="num">{coef}{st}</td>
          <td class="num">{se}</td>
          <td class="num">{pv}</td>
          <td class="num ci">{ci}</td>
          <td class="num">{n}</td>
          <td class="num">{bw}</td>
        </tr>\n"""

    return f"""    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Controls</th>
          <th>Coef.</th>
          <th>Std. Err.</th>
          <th><i>p</i>-value</th>
          <th>95% CI</th>
          <th><i>N</i></th>
          <th>BW</th>
        </tr>
      </thead>
      <tbody>
{rows_html}      </tbody>
    </table>"""


# ── HTML assembly ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RDD Results: Rotten Tomatoes &amp; Box Office Revenue</title>
<style>
  :root {{
    --bg: #fff;
    --fg: #1a1a1a;
    --muted: #6b7280;
    --accent: #4a90d9;
    --border: #e5e7eb;
    --row-alt: #f9fafb;
    --preferred-bg: #eff6ff;
    --star-color: #b45309;
  }}
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: var(--fg);
    background: var(--bg);
    line-height: 1.6;
    margin: 0;
    padding: 2rem 1rem;
  }}
  .container {{
    max-width: 960px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 0.25rem;
    letter-spacing: -0.01em;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 2rem;
  }}
  h2 {{
    font-size: 1.25rem;
    font-weight: 600;
    margin: 2.5rem 0 0.75rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid var(--border);
  }}
  h3 {{
    font-size: 1.05rem;
    font-weight: 600;
    margin: 1.5rem 0 0.5rem;
    color: var(--fg);
  }}
  p, li {{
    font-size: 0.925rem;
    color: #374151;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 0.75rem 0 1.5rem;
  }}
  thead th {{
    background: #f3f4f6;
    font-weight: 600;
    text-align: left;
    padding: 0.5rem 0.6rem;
    border-bottom: 2px solid var(--border);
    font-size: 0.8rem;
    white-space: nowrap;
  }}
  tbody td {{
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid var(--border);
  }}
  tbody tr:nth-child(even) {{ background: var(--row-alt); }}
  tbody tr.preferred {{
    background: var(--preferred-bg);
    font-weight: 600;
  }}
  .num {{
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    text-align: right;
    font-size: 0.82rem;
    white-space: nowrap;
  }}
  .ci {{ font-size: 0.78rem; }}
  .stars {{
    color: var(--star-color);
    font-weight: 700;
    margin-left: 2px;
  }}
  .plot-row {{
    display: flex;
    gap: 1.25rem;
    flex-wrap: wrap;
    margin: 1rem 0;
  }}
  .plot-row img {{
    flex: 1 1 300px;
    max-width: 100%;
    height: auto;
    border: 1px solid var(--border);
    border-radius: 4px;
  }}
  .plot-single {{
    margin: 1rem 0;
  }}
  .plot-single img {{
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 4px;
  }}
  .notes {{
    font-size: 0.8rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    padding-top: 1rem;
    margin-top: 2rem;
  }}
  .notes p {{
    font-size: 0.8rem;
    color: var(--muted);
    margin: 0.25rem 0;
  }}
  .findings li {{
    margin-bottom: 0.5rem;
  }}
  @media (max-width: 640px) {{
    table {{ font-size: 0.75rem; }}
    .num {{ font-size: 0.73rem; }}
    .plot-row {{ flex-direction: column; }}
  }}
</style>
</head>
<body>
<div class="container">

<h1>Does anyone else care if the movie&rsquo;s &ldquo;Fresh?&rdquo;</h1>
<p class="subtitle">
  A regression discontinuity side project investigating whether Rotten Tomatoes scores have a causal effect on box office revenue in the post-COVID era.
  Data and code available on <a href="https://github.com/grantbw4/Regression_Discontinuity_RT" style="color: var(--accent);">GitHub</a>.
</p>

<h2>Motivation</h2>
<p>
  Fun fact about me: I love reviews. I have a near-religious obsession with online review aggregators:
  Rotten Tomatoes, Pitchfork, Metacritic, Goodreads, you name it. I write a Google Review for every
  restaurant I visit. I&rsquo;m not entirely sure why I value strangers&rsquo; opinions so much, but
  that&rsquo;s a blog post for another day.
</p>
<p>
  So when I came across a
  <a href="https://www.hollywoodreporter.com/movies/movie-news/melania-movie-rotten-tomatoes-audience-score-critics-1236497256/">Hollywood Reporter article</a>
  about the <i>Melania</i> documentary holding the record for the largest critic&ndash;audience
  score gap in Rotten Tomatoes history (6% critics vs. 99% audience), I was intrigued. It got me
  wondering: do people actually consult Rotten Tomatoes before buying a ticket? And could you
  identify these scores having any real, causal effect on revenue?
</p>
<p>
  The &ldquo;Fresh&rdquo; vs. &ldquo;Rotten&rdquo; label creates a sharp cutoff at 60%, which
  screamed regression discontinuity to me. A quick Google search confirmed I wasn&rsquo;t the first
  to think of this: Nishijima, Rodrigues &amp; Souza
  (<a href="https://www.tandfonline.com/doi/full/10.1080/13504851.2021.1918324">2021</a>)
  ran an RDD on the Tomatometer using 1,239 films from 1999&ndash;2019 and found no effect.
  Their paper was interesting, but I wanted to see if things looked different in the post-COVID era,
  when theatrical appetite has arguably shrunk and audiences might be more selective about what&rsquo;s
  worth a trip to the theater. I also wanted to extend the analysis to the <b>Audience Score</b>,
  which their study didn&rsquo;t cover.
</p>

<h2>Approach</h2>
<p>
  Rotten Tomatoes labels a film &ldquo;Fresh&rdquo; if its score is &ge;&nbsp;60% and
  &ldquo;Rotten&rdquo; if it&rsquo;s &lt;&nbsp;60%. If the binary label itself nudges people to
  buy tickets, we&rsquo;d expect a discontinuous jump in box office revenue right at that cutoff,
  beyond what the continuous score would predict. I test this for both the <b>Tomatometer</b>
  (critic consensus) and the <b>Audience Score</b>, using two outcomes: log opening weekend gross
  and log total domestic gross.
</p>
<p>
  The preferred estimates use <b>rdrobust</b> (Calonico, Cattaneo &amp; Titiunik, 2014) with
  MSE-optimal bandwidth and a triangular kernel. I also run parametric OLS with linear and quadratic
  polynomials (interacted with treatment) as robustness checks. Standard errors are
  heteroskedasticity-robust (HC1). Controls include log budget, log opening theaters,
  MPAA rating dummies, and release-year fixed effects.
</p>

<h2>Data</h2>
<p>
  I scraped three sources and merged them together:
</p>
<ul>
  <li><b>Box Office Mojo</b> for the core film list and box office numbers: opening weekend gross,
  theater counts, total domestic gross, MPAA rating, and release date. I pulled every wide release
  (600+ theaters) from September 2021 through February 2026.</li>
  <li><b>Rotten Tomatoes</b> for the running variables: Tomatometer (critic consensus) and Audience Score,
  plus review/rating counts. I matched each film by constructing URL slugs from the title, falling back
  to RT&rsquo;s search page when that didn&rsquo;t work.</li>
  <li><b>The Numbers</b> for production budgets, which I fuzzy-matched to the BOM titles by normalized
  name and release year.</li>
</ul>
<p>
  After merging and filtering, the final dataset has <b>621 films</b>. Films still in theatrical release
  at the end of the study window are flagged and excluded from the total domestic gross analysis
  (since their grosses are incomplete).
</p>

<h2>Score Distributions</h2>
<p>
  Before diving into results, a sanity check: for an RDD to work, the density of scores should be
  smooth through the 60% cutoff. If a bunch of films were suspiciously clustered just above it,
  that would suggest some kind of manipulation. Things look clean here.
</p>
<div class="plot-row">
  <img src="data:image/png;base64,{density_critic}" alt="Tomatometer distribution">
  <img src="data:image/png;base64,{density_audience}" alt="Audience Score distribution">
</div>

<h2>Results</h2>

<h3>Panel A: Critic Score (Tomatometer)</h3>
<p><b>Outcome: Log Opening Weekend Gross</b></p>
{table_critic_opening}

<p><b>Outcome: Log Total Domestic Gross</b> <span style="color:var(--muted)">(excluding films still in theaters)</span></p>
{table_critic_total}

<h3>Panel B: Audience Score</h3>
<p><b>Outcome: Log Opening Weekend Gross</b></p>
{table_audience_opening}

<p><b>Outcome: Log Total Domestic Gross</b> <span style="color:var(--muted)">(excluding films still in theaters)</span></p>
{table_audience_total}

<h2>RDD Plots</h2>
<p>Binned scatter plots with quadratic fits on each side of the cutoff. If the &ldquo;Fresh&rdquo;
label were causing a jump in revenue, you&rsquo;d see a visible gap at zero. Spoiler: you don&rsquo;t.</p>

<div class="plot-row">
  <img src="data:image/png;base64,{scatter_critic_opening}" alt="Critic - Opening Gross">
  <img src="data:image/png;base64,{scatter_critic_total}" alt="Critic - Total Gross">
</div>
<div class="plot-row">
  <img src="data:image/png;base64,{scatter_audience_opening}" alt="Audience - Opening Gross">
  <img src="data:image/png;base64,{scatter_audience_total}" alt="Audience - Total Gross">
</div>

<h2>What I Found</h2>
<ul class="findings">
  <li><b>No discontinuity at the critic threshold.</b> The rdrobust estimate for opening gross is
  0.060 log points (<i>p</i>&nbsp;=&nbsp;0.82) with controls, statistically and economically
  indistinguishable from zero. Total gross estimates are similarly null. Just like Nishijima et al.,
  the Tomatometer label doesn&rsquo;t seem to matter.</li>
  <li><b>No discontinuity at the audience threshold either.</b> The preferred estimate is &minus;0.064
  log points (<i>p</i>&nbsp;=&nbsp;0.84) for opening gross. A na&iuml;ve OLS linear spec looks
  significant, but that vanishes once you add controls or allow for a quadratic. Classic
  functional form misspecification rather than a real discontinuity.</li>
  <li><b>Bottom line:</b> even in the post-COVID era, crossing the 60% &ldquo;Fresh&rdquo; threshold
  on Rotten Tomatoes doesn&rsquo;t appear to cause a detectable bump in box office revenue. People
  might respond to the underlying score or individual reviews, but the binary label itself?
  Not so much, or at least not enough to pick up in 621 films.</li>
</ul>

<div class="notes">
  <p><b>Notes:</b> Shaded rows are the preferred spec (rdrobust with controls).
  *, **, *** = significance at 10%, 5%, 1%.
  rdrobust reports robust bias-corrected coefficients and CIs;
  <i>N</i> is the effective sample within the MSE-optimal bandwidth (BW).
  OLS uses the full score range.
  Coefficients are in log points (multiply by 100 for approximate % effect).</p>
  <p><b>References:</b><br>
  Calonico, S., Cattaneo, M.D. &amp; Titiunik, R. (2014).
  &ldquo;Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs.&rdquo;
  <i>Econometrica</i>, 82(6), 2295&ndash;2326.<br>
  Nishijima, M., Rodrigues, M. &amp; Souza, T.L.D. (2021).
  &ldquo;Is Rotten Tomatoes killing the movie industry? A regression discontinuity approach.&rdquo;
  <i>Applied Economics Letters</i>, 29(13), 1187&ndash;1192.</p>
</div>

</div>
</body>
</html>
"""


def main():
    df = pd.read_csv(config.DATA_PROCESSED / "merged_dataset.csv")
    results = pd.read_csv(config.PROJECT_ROOT / "output" / "rdd_results_raw.csv")

    print("Generating plots...")

    # Density plots
    density_critic = plot_density(df, "tomatometer", "Tomatometer")
    density_audience = plot_density(df, "audience_score", "Audience Score")

    # RDD scatter plots (exclude in_theaters for total gross)
    df_complete = df[df["in_theaters"] == False]

    scatter_critic_opening = plot_rdd_scatter(
        df, "tomatometer_centered", "log_opening_gross",
        "Tomatometer", "Log Opening Gross")
    scatter_critic_total = plot_rdd_scatter(
        df_complete, "tomatometer_centered", "log_total_gross",
        "Tomatometer", "Log Total Gross")
    scatter_audience_opening = plot_rdd_scatter(
        df, "audience_score_centered", "log_opening_gross",
        "Audience Score", "Log Opening Gross")
    scatter_audience_total = plot_rdd_scatter(
        df_complete, "audience_score_centered", "log_total_gross",
        "Audience Score", "Log Total Gross")

    print("Building tables...")

    table_critic_opening = build_results_table(results, "Critic", "Log Opening Gross")
    table_critic_total = build_results_table(results, "Critic", "Log Total Gross")
    table_audience_opening = build_results_table(results, "Audience", "Log Opening Gross")
    table_audience_total = build_results_table(results, "Audience", "Log Total Gross")

    print("Assembling HTML...")

    html = HTML_TEMPLATE.format(
        density_critic=density_critic,
        density_audience=density_audience,
        scatter_critic_opening=scatter_critic_opening,
        scatter_critic_total=scatter_critic_total,
        scatter_audience_opening=scatter_audience_opening,
        scatter_audience_total=scatter_audience_total,
        table_critic_opening=table_critic_opening,
        table_critic_total=table_critic_total,
        table_audience_opening=table_audience_opening,
        table_audience_total=table_audience_total,
    )

    output_path = config.PROJECT_ROOT / "output" / "rdd_report.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {output_path}")

    docs_path = config.PROJECT_ROOT / "docs" / "index.html"
    docs_path.parent.mkdir(exist_ok=True)
    docs_path.write_text(html, encoding="utf-8")
    print(f"GitHub Pages copy saved to {docs_path}")


if __name__ == "__main__":
    main()
