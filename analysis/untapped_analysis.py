#!/usr/bin/env python3
"""
Untapped Cache Analysis
========================
Extracts 5 additional dimensions from the already-cached GitHub stats/contributors
JSON files — no new API calls required.

  1. Code-volume concentration (additions + deletions vs commits)
  2. Contributor tenure & longevity
  3. Arrival / departure curves (monthly new & departing contributors)
  4. Activity consistency (fraction of active weeks)
  5. Authorship vs maintenance (additions-heavy vs deletions-heavy contributors)

Generates CSVs, summary statistics, and charts.
"""

import json, warnings, datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "output"
CACHE_DIR  = SCRIPT_DIR / "cache"

summary = pd.read_csv(OUT_DIR / "centralization_summary.csv")
sample  = pd.read_csv(OUT_DIR / "sampled_repos.csv")

# ── Helpers ──────────────────────────────────────────────────
def gini(arr):
    a = np.sort(np.array(arr, dtype=float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * a) - (n + 1) * np.sum(a)) / (n * np.sum(a)))

def hhi(arr):
    a = np.array(arr, dtype=float)
    total = a.sum()
    if total == 0:
        return np.nan
    shares = a / total
    return float(np.sum(shares ** 2))

def top_k_share(arr, k):
    a = np.sort(np.array(arr, dtype=float))[::-1]
    total = a.sum()
    if total == 0:
        return np.nan
    return float(a[:k].sum() / total)

def ts_to_date(ts):
    return datetime.datetime.utcfromtimestamp(ts)


# ==============================================================
# PASS 1: Parse all cache files
# ==============================================================
print("Parsing cached contributor data...")

repo_records = []          # per-repo level metrics
contributor_records = []   # per-contributor level records
monthly_arrivals = defaultdict(lambda: defaultdict(set))   # repo -> month -> set(logins)
global_dev_first = {}      # login -> earliest active timestamp
global_dev_last  = {}      # login -> latest active timestamp

for cache_file in sorted(CACHE_DIR.glob("*.stats_contributors.json")):
    repo_name = cache_file.stem.replace(".stats_contributors", "").replace("__", "/")

    with open(cache_file) as f:
        stats = json.load(f)

    if not stats:
        continue

    # Per-contributor arrays for this repo
    commits_arr = []
    additions_arr = []
    deletions_arr = []
    loc_arr = []          # additions + deletions (total code churn)
    tenure_weeks_arr = []
    active_weeks_arr = []
    total_weeks_arr = []
    first_active_arr = []
    last_active_arr = []
    net_additions_arr = []  # additions - deletions (net code growth)

    for contrib in stats:
        login = (contrib.get("author") or {}).get("login")
        if not login:
            continue

        weeks = contrib.get("weeks", [])
        if not weeks:
            continue

        total_commits = contrib.get("total", 0)
        total_add = sum(w.get("a", 0) for w in weeks)
        total_del = sum(w.get("d", 0) for w in weeks)
        total_loc = total_add + total_del

        # Active weeks: weeks where c > 0 or a > 0 or d > 0
        active_wks = [w for w in weeks if w.get("c", 0) > 0 or w.get("a", 0) > 0 or w.get("d", 0) > 0]
        n_active = len(active_wks)
        n_total = len(weeks)

        if n_active == 0:
            continue

        # Tenure: first active week to last active week
        active_ts = sorted([w["w"] for w in active_wks])
        first_ts = active_ts[0]
        last_ts = active_ts[-1]
        tenure_weeks = max(1, (last_ts - first_ts) // (7 * 86400) + 1)

        # Consistency: fraction of weeks between first and last that were active
        span_weeks = max(1, (last_ts - first_ts) // (7 * 86400) + 1)
        consistency = n_active / span_weeks if span_weeks > 0 else 0.0

        commits_arr.append(total_commits)
        additions_arr.append(total_add)
        deletions_arr.append(total_del)
        loc_arr.append(total_loc)
        tenure_weeks_arr.append(tenure_weeks)
        active_weeks_arr.append(n_active)
        total_weeks_arr.append(n_total)
        first_active_arr.append(first_ts)
        last_active_arr.append(last_ts)
        net_additions_arr.append(total_add - total_del)

        # Track monthly arrival per repo
        first_month = ts_to_date(first_ts).strftime("%Y-%m")
        last_month = ts_to_date(last_ts).strftime("%Y-%m")
        monthly_arrivals[repo_name][first_month].add(login)

        # Global developer tracking
        if login not in global_dev_first or first_ts < global_dev_first[login]:
            global_dev_first[login] = first_ts
        if login not in global_dev_last or last_ts > global_dev_last[login]:
            global_dev_last[login] = last_ts

        # Classify: author (net positive) vs maintainer (net negative or balanced)
        if total_loc > 0:
            add_ratio = total_add / total_loc
        else:
            add_ratio = 0.5

        contributor_records.append({
            "owner_repo": repo_name,
            "login": login,
            "commits": total_commits,
            "additions": total_add,
            "deletions": total_del,
            "loc_churn": total_loc,
            "net_additions": total_add - total_del,
            "tenure_weeks": tenure_weeks,
            "active_weeks": n_active,
            "span_weeks": span_weeks,
            "consistency": consistency,
            "addition_ratio": add_ratio,
            "first_active_ts": first_ts,
            "last_active_ts": last_ts,
        })

    if len(commits_arr) < 2:
        continue

    # ── Per-repo metrics ─────────────────────────────────────
    commits_arr = np.array(commits_arr)
    loc_arr = np.array(loc_arr)
    additions_arr = np.array(additions_arr)
    deletions_arr = np.array(deletions_arr)

    # Code-volume concentration
    gini_loc = gini(loc_arr)
    hhi_loc = hhi(loc_arr)
    top1_loc = top_k_share(loc_arr, 1)

    # Additions-only concentration (pure authorship)
    gini_add = gini(additions_arr)
    gini_del = gini(deletions_arr)

    # Tenure stats
    med_tenure = np.median(tenure_weeks_arr)
    max_tenure = max(tenure_weeks_arr)
    med_consistency = np.median([c["consistency"] for c in contributor_records
                                  if c["owner_repo"] == repo_name])

    # Contributor half-life: what fraction have been active in the last 26 weeks?
    cutoff_ts = max(last_active_arr) - 26 * 7 * 86400  # 6 months before last activity
    recently_active = sum(1 for t in last_active_arr if t >= cutoff_ts)
    retention_6m = recently_active / len(last_active_arr)

    # One-time contributors: active_weeks == 1
    one_timers = sum(1 for a in active_weeks_arr if a <= 1)
    one_timer_frac = one_timers / len(active_weeks_arr)

    repo_records.append({
        "owner_repo": repo_name,
        "n_contributors": len(commits_arr),
        "gini_commits": gini(commits_arr),
        "gini_loc": gini_loc,
        "hhi_loc": hhi_loc,
        "top1_share_loc": top1_loc,
        "gini_additions": gini_add,
        "gini_deletions": gini_del,
        "total_additions": int(additions_arr.sum()),
        "total_deletions": int(deletions_arr.sum()),
        "total_loc_churn": int(loc_arr.sum()),
        "median_tenure_weeks": med_tenure,
        "max_tenure_weeks": max_tenure,
        "median_consistency": med_consistency,
        "retention_6m": retention_6m,
        "one_timer_fraction": one_timer_frac,
    })

repo_df = pd.DataFrame(repo_records)
contrib_df = pd.DataFrame(contributor_records)

print(f"  {len(repo_df)} repos, {len(contrib_df)} contributor-repo pairs")

# Merge with sample metadata
repo_df = repo_df.merge(
    sample[["owner_repo", "_language", "_age_weeks", "_stars", "category"]],
    on="owner_repo", how="left"
)

repo_df.to_csv(OUT_DIR / "untapped_repo_metrics.csv", index=False)
contrib_df.to_csv(OUT_DIR / "untapped_contributor_details.csv", index=False)


# ==============================================================
# ANALYSIS 1: Code-Volume Concentration
# ==============================================================
print("\n" + "=" * 65)
print("  ANALYSIS 1: Code-Volume Concentration (LoC vs Commits)")
print("=" * 65)

print(f"\n  {'Metric':<30s} {'Median':>9s} {'Mean':>9s} {'Std':>9s}")
print(f"  {'-'*57}")
for col, label in [
    ("gini_commits", "Gini (commits)"),
    ("gini_loc", "Gini (LoC churn)"),
    ("gini_additions", "Gini (additions only)"),
    ("gini_deletions", "Gini (deletions only)"),
    ("hhi_loc", "HHI (LoC churn)"),
    ("top1_share_loc", "Top-1 share (LoC)"),
]:
    vals = repo_df[col].dropna()
    print(f"  {label:<30s} {vals.median():>9.3f} {vals.mean():>9.3f} {vals.std():>9.3f}")

# Compare commit Gini vs LoC Gini
valid = repo_df[["gini_commits", "gini_loc"]].dropna()
rho, p = sp_stats.spearmanr(valid["gini_commits"], valid["gini_loc"])
print(f"\n  Spearman(Gini_commits, Gini_LoC): rho={rho:.3f}, p={p:.6f}")

diff = valid["gini_loc"] - valid["gini_commits"]
print(f"  Gini_LoC - Gini_commits: median={diff.median():.3f}, mean={diff.mean():.3f}")
print(f"  Repos where LoC Gini > commit Gini: {(diff > 0).sum()} / {len(diff)} "
      f"({100*(diff>0).mean():.1f}%)")

# Wilcoxon signed-rank test
w_stat, w_p = sp_stats.wilcoxon(valid["gini_loc"], valid["gini_commits"])
print(f"  Wilcoxon signed-rank: W={w_stat:.0f}, p={w_p:.6f} "
      f"{'***' if w_p < 0.001 else '**' if w_p < 0.01 else '*' if w_p < 0.05 else 'ns'}")


# ==============================================================
# ANALYSIS 2: Contributor Tenure & Longevity
# ==============================================================
print("\n" + "=" * 65)
print("  ANALYSIS 2: Contributor Tenure & Longevity")
print("=" * 65)

print(f"\n  Per-repo tenure statistics:")
print(f"  {'Metric':<30s} {'Median':>9s} {'Mean':>9s} {'P90':>9s}")
print(f"  {'-'*57}")
for col, label in [
    ("median_tenure_weeks", "Med contributor tenure (wks)"),
    ("max_tenure_weeks", "Max contributor tenure (wks)"),
    ("retention_6m", "6-month retention rate"),
    ("one_timer_fraction", "One-timer fraction"),
]:
    vals = repo_df[col].dropna()
    print(f"  {label:<30s} {vals.median():>9.2f} {vals.mean():>9.2f} "
          f"{vals.quantile(0.9):>9.2f}")

# Per-contributor tenure
print(f"\n  Per-contributor tenure:")
print(f"    Median tenure:     {contrib_df['tenure_weeks'].median():.0f} weeks")
print(f"    Mean tenure:       {contrib_df['tenure_weeks'].mean():.1f} weeks")
print(f"    P90 tenure:        {contrib_df['tenure_weeks'].quantile(0.9):.0f} weeks")
print(f"    One-week only:     {(contrib_df['active_weeks'] == 1).sum()} / "
      f"{len(contrib_df)} ({100*(contrib_df['active_weeks']==1).mean():.1f}%)")

# Tenure vs centralization correlation
for metric in ["gini_commits", "gini_loc", "retention_6m", "one_timer_fraction"]:
    valid = repo_df[["median_tenure_weeks", metric]].dropna()
    rho, p = sp_stats.spearmanr(valid["median_tenure_weeks"], valid[metric])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  Spearman(med_tenure, {metric:<22s}): rho={rho:+.3f}, p={p:.4f} {sig}")


# ==============================================================
# ANALYSIS 3: Arrival & Departure Curves
# ==============================================================
print("\n" + "=" * 65)
print("  ANALYSIS 3: Contributor Arrival & Departure Trends")
print("=" * 65)

# Global arrival/departure by month
all_first_months = defaultdict(int)
all_last_months = defaultdict(int)
for _, row in contrib_df.iterrows():
    fm = ts_to_date(row["first_active_ts"]).strftime("%Y-%m")
    lm = ts_to_date(row["last_active_ts"]).strftime("%Y-%m")
    all_first_months[fm] += 1
    all_last_months[lm] += 1

months = sorted(set(all_first_months.keys()) | set(all_last_months.keys()))
arrival_series = pd.Series({m: all_first_months.get(m, 0) for m in months})
departure_series = pd.Series({m: all_last_months.get(m, 0) for m in months})

# Only show meaningful range (skip near-empty early months)
min_month = arrival_series[arrival_series > 5].index[0] if (arrival_series > 5).any() else months[0]
arrival_series = arrival_series[arrival_series.index >= min_month]
departure_series = departure_series[departure_series.index >= min_month]

# Net change
net_series = arrival_series.subtract(departure_series, fill_value=0)

print(f"  Monthly stats (from {min_month}):")
print(f"    Peak arrivals:    {arrival_series.max()} ({arrival_series.idxmax()})")
print(f"    Peak departures:  {departure_series.max()} ({departure_series.idxmax()})")
print(f"    Recent 12 months: avg arrivals={arrival_series.tail(12).mean():.0f}/mo, "
      f"avg departures={departure_series.tail(12).mean():.0f}/mo")

# Cumulative active contributors over time
cumulative = (arrival_series.cumsum() - departure_series.reindex(arrival_series.index, fill_value=0).cumsum())


# ==============================================================
# ANALYSIS 4: Activity Consistency
# ==============================================================
print("\n" + "=" * 65)
print("  ANALYSIS 4: Activity Consistency")
print("=" * 65)

print(f"\n  Per-contributor consistency (active weeks / span weeks):")
print(f"    Median: {contrib_df['consistency'].median():.3f}")
print(f"    Mean:   {contrib_df['consistency'].mean():.3f}")
print(f"    <10%:   {(contrib_df['consistency'] < 0.1).sum()} "
      f"({100*(contrib_df['consistency'] < 0.1).mean():.1f}%)")
print(f"    >50%:   {(contrib_df['consistency'] > 0.5).sum()} "
      f"({100*(contrib_df['consistency'] > 0.5).mean():.1f}%)")
print(f"    100%:   {(contrib_df['consistency'] >= 0.99).sum()} "
      f"({100*(contrib_df['consistency'] >= 0.99).mean():.1f}%)")

# Consistency vs contribution volume
high_consist = contrib_df[contrib_df["consistency"] > 0.5]
low_consist = contrib_df[contrib_df["consistency"] <= 0.1]
print(f"\n  High-consistency devs (>50%): {len(high_consist)}")
print(f"    Median commits: {high_consist['commits'].median():.0f}, "
      f"median LoC: {high_consist['loc_churn'].median():.0f}")
print(f"  Low-consistency devs (≤10%): {len(low_consist)}")
print(f"    Median commits: {low_consist['commits'].median():.0f}, "
      f"median LoC: {low_consist['loc_churn'].median():.0f}")

# Repo-level: correlation of median consistency with centralization
for metric in ["gini_commits", "gini_loc", "one_timer_fraction"]:
    valid = repo_df[["median_consistency", metric]].dropna()
    rho, p = sp_stats.spearmanr(valid["median_consistency"], valid[metric])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  Spearman(consistency, {metric:<22s}): rho={rho:+.3f}, p={p:.4f} {sig}")


# ==============================================================
# ANALYSIS 5: Authorship vs Maintenance Profiles
# ==============================================================
print("\n" + "=" * 65)
print("  ANALYSIS 5: Authorship vs Maintenance")
print("=" * 65)

# Classify contributors
# addition_ratio > 0.7 → mostly authoring new code
# addition_ratio < 0.3 → mostly maintaining/refactoring (deleting > adding)
# 0.3-0.7 → balanced
contrib_df["role"] = pd.cut(
    contrib_df["addition_ratio"],
    bins=[0, 0.3, 0.7, 1.0],
    labels=["Maintainer", "Balanced", "Author"],
    include_lowest=True
)

role_stats = contrib_df.groupby("role", observed=True).agg(
    n=("commits", "count"),
    med_commits=("commits", "median"),
    med_loc=("loc_churn", "median"),
    med_tenure=("tenure_weeks", "median"),
    med_consistency=("consistency", "median"),
).reset_index()

print(f"\n  Contributor Role Classification (by addition ratio):")
print(f"  {'Role':<14s} {'n':>6s} {'%':>7s} {'Med Commits':>12s} {'Med LoC':>9s} "
      f"{'Med Tenure':>11s} {'Med Consist':>12s}")
print(f"  {'-'*72}")
for _, row in role_stats.iterrows():
    pct = 100 * row["n"] / len(contrib_df)
    print(f"  {row['role']:<14s} {row['n']:>6.0f} {pct:>6.1f}% {row['med_commits']:>12.0f} "
          f"{row['med_loc']:>9.0f} {row['med_tenure']:>11.0f} {row['med_consistency']:>12.3f}")

# Per-repo: is the top contributor an author or maintainer?
top_contribs = contrib_df.sort_values(["owner_repo", "commits"], ascending=[True, False])
top_contribs = top_contribs.groupby("owner_repo").first().reset_index()
print(f"\n  Top contributor per repo — role distribution:")
top_role_counts = top_contribs["role"].value_counts()
for role, count in top_role_counts.items():
    print(f"    {role}: {count} repos ({100*count/len(top_contribs):.1f}%)")

# Addition concentration vs deletion concentration
valid = repo_df[["gini_additions", "gini_deletions"]].dropna()
rho, p = sp_stats.spearmanr(valid["gini_additions"], valid["gini_deletions"])
print(f"\n  Spearman(Gini_additions, Gini_deletions): rho={rho:.3f}, p={p:.6f}")
diff_ad = valid["gini_additions"] - valid["gini_deletions"]
print(f"  Gini_add - Gini_del: median={diff_ad.median():.3f}, "
      f"repos where add Gini > del Gini: {(diff_ad>0).sum()}/{len(diff_ad)} "
      f"({100*(diff_ad>0).mean():.1f}%)")


# ==============================================================
# CHARTS
# ==============================================================
print("\n" + "=" * 65)
print("  Generating charts...")
print("=" * 65)

# ── Chart 1: Commit Gini vs LoC Gini scatter ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Code-Volume Concentration: Commits vs Lines-of-Code",
             fontsize=14, fontweight="bold")

ax = axes[0]
ax.scatter(repo_df["gini_commits"], repo_df["gini_loc"], alpha=0.4, s=20, c="#4C72B0")
ax.plot([0, 1], [0, 1], "r--", lw=1.5, alpha=0.6, label="y = x (equal)")
ax.set_xlabel("Gini (commits)"); ax.set_ylabel("Gini (LoC churn)")
ax.set_title("Commit vs LoC Concentration")
ax.legend(fontsize=9)

# Histogram of the difference
ax = axes[1]
diff = (repo_df["gini_loc"] - repo_df["gini_commits"]).dropna()
ax.hist(diff, bins=30, edgecolor="black", alpha=0.7, color="#55A868")
ax.axvline(0, color="red", ls="--", lw=1.5, label="No difference")
ax.axvline(diff.median(), color="blue", ls="--", lw=1.5,
           label=f"Median = {diff.median():+.3f}")
ax.set_xlabel("Gini(LoC) − Gini(commits)")
ax.set_ylabel("Repos")
ax.set_title("LoC Concentration Premium")
ax.legend(fontsize=9)

# Additions vs Deletions Gini
ax = axes[2]
ax.scatter(repo_df["gini_additions"], repo_df["gini_deletions"],
           alpha=0.4, s=20, c="#C44E52")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("Gini (additions)"); ax.set_ylabel("Gini (deletions)")
ax.set_title("Authorship vs Maintenance Concentration")

plt.tight_layout()
plt.savefig(OUT_DIR / "loc_concentration.png", dpi=150)
plt.close()
print("  Saved: loc_concentration.png")


# ── Chart 2: Tenure & Retention ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Contributor Tenure & Retention", fontsize=14, fontweight="bold")

# Contributor tenure distribution
ax = axes[0, 0]
tenure = contrib_df["tenure_weeks"].clip(upper=300)
ax.hist(tenure, bins=50, edgecolor="black", alpha=0.7, color="#4C72B0", log=True)
ax.axvline(tenure.median(), color="red", ls="--",
           label=f"Median = {contrib_df['tenure_weeks'].median():.0f} wks")
ax.set_xlabel("Tenure (weeks)"); ax.set_ylabel("Contributors (log)")
ax.set_title("Contributor Tenure Distribution"); ax.legend()

# One-timer fraction distribution
ax = axes[0, 1]
ax.hist(repo_df["one_timer_fraction"].dropna(), bins=25, edgecolor="black",
        alpha=0.7, color="#DD8452")
ax.axvline(repo_df["one_timer_fraction"].median(), color="red", ls="--",
           label=f"Median = {repo_df['one_timer_fraction'].median():.2f}")
ax.set_xlabel("One-Timer Fraction"); ax.set_ylabel("Repos")
ax.set_title("Fraction of One-Time Contributors per Repo"); ax.legend()

# Retention vs Gini
ax = axes[1, 0]
ax.scatter(repo_df["retention_6m"], repo_df["gini_commits"],
           alpha=0.4, s=20, c="#55A868")
rho, p = sp_stats.spearmanr(
    repo_df[["retention_6m","gini_commits"]].dropna()["retention_6m"],
    repo_df[["retention_6m","gini_commits"]].dropna()["gini_commits"])
ax.set_xlabel("6-Month Retention Rate"); ax.set_ylabel("Gini (commits)")
ax.set_title(f"Retention vs Centralization (ρ={rho:.2f})")

# Median tenure vs n_contributors
ax = axes[1, 1]
merged_t = repo_df.merge(summary[["owner_repo", "n_authors"]], on="owner_repo")
ax.scatter(merged_t["n_contributors"], merged_t["median_tenure_weeks"],
           alpha=0.4, s=20, c="#8172B2")
ax.set_xlabel("Number of Contributors"); ax.set_ylabel("Median Tenure (weeks)")
ax.set_title("Team Size vs Contributor Tenure")
ax.set_xscale("log")

plt.tight_layout()
plt.savefig(OUT_DIR / "tenure_retention.png", dpi=150)
plt.close()
print("  Saved: tenure_retention.png")


# ── Chart 3: Arrival/Departure curves ────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
fig.suptitle("Contributor Arrival & Departure Over Time (All Repos)",
             fontsize=14, fontweight="bold")

# Use quarterly rolling average for smoother curve
ax = axes[0]
arr_q = arrival_series.rolling(3, min_periods=1).mean()
dep_q = departure_series.reindex(arrival_series.index, fill_value=0).rolling(3, min_periods=1).mean()

x_ticks = range(0, len(arrival_series), max(1, len(arrival_series)//20))
x_labels = [arrival_series.index[i] for i in x_ticks]

ax.fill_between(range(len(arr_q)), arr_q, alpha=0.3, color="#55A868")
ax.plot(range(len(arr_q)), arr_q, color="#55A868", lw=2, label="New contributors (3-mo avg)")
ax.fill_between(range(len(dep_q)), dep_q, alpha=0.3, color="#C44E52")
ax.plot(range(len(dep_q)), dep_q, color="#C44E52", lw=2, label="Departing contributors (3-mo avg)")
ax.set_ylabel("Contributors / Month"); ax.legend(fontsize=10)
ax.set_title("Monthly New vs Departing Contributors")
ax.set_xticks(list(x_ticks))
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

# Net flow
ax = axes[1]
net = arrival_series.subtract(departure_series.reindex(arrival_series.index, fill_value=0))
net_q = net.rolling(3, min_periods=1).mean()
colors = ["#55A868" if v >= 0 else "#C44E52" for v in net_q]
ax.bar(range(len(net_q)), net_q, color=colors, alpha=0.7, width=1.0)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Net Contributors / Month")
ax.set_title("Net Contributor Flow (Arrivals − Departures)")
ax.set_xticks(list(x_ticks))
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "arrival_departure.png", dpi=150)
plt.close()
print("  Saved: arrival_departure.png")


# ── Chart 4: Consistency analysis ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Activity Consistency Analysis", fontsize=14, fontweight="bold")

# Distribution
ax = axes[0]
ax.hist(contrib_df["consistency"].dropna(), bins=30, edgecolor="black",
        alpha=0.7, color="#4C72B0")
ax.axvline(contrib_df["consistency"].median(), color="red", ls="--",
           label=f"Median = {contrib_df['consistency'].median():.2f}")
ax.set_xlabel("Consistency (active weeks / span)")
ax.set_ylabel("Contributors"); ax.set_title("Consistency Distribution"); ax.legend()

# Consistency vs commits (contributor level)
ax = axes[1]
ax.scatter(contrib_df["consistency"], contrib_df["commits"].clip(upper=5000),
           alpha=0.15, s=5, c="#55A868")
ax.set_xlabel("Consistency"); ax.set_ylabel("Commits (capped at 5000)")
ax.set_title("Consistency vs Output Volume")

# Repo-level consistency vs Gini
ax = axes[2]
ax.scatter(repo_df["median_consistency"], repo_df["gini_loc"],
           alpha=0.4, s=20, c="#C44E52")
rho, p = sp_stats.spearmanr(
    repo_df[["median_consistency","gini_loc"]].dropna()["median_consistency"],
    repo_df[["median_consistency","gini_loc"]].dropna()["gini_loc"])
ax.set_xlabel("Median Consistency"); ax.set_ylabel("Gini (LoC)")
ax.set_title(f"Team Consistency vs LoC Concentration (ρ={rho:.2f})")

plt.tight_layout()
plt.savefig(OUT_DIR / "consistency_analysis.png", dpi=150)
plt.close()
print("  Saved: consistency_analysis.png")


# ── Chart 5: Authorship vs Maintenance roles ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Authorship vs Maintenance Contributor Profiles",
             fontsize=14, fontweight="bold")

# Addition ratio distribution
ax = axes[0]
ax.hist(contrib_df["addition_ratio"].dropna(), bins=40, edgecolor="black",
        alpha=0.7, color="#4C72B0")
ax.axvline(0.3, color="red", ls="--", alpha=0.7, label="Maintainer threshold")
ax.axvline(0.7, color="green", ls="--", alpha=0.7, label="Author threshold")
ax.set_xlabel("Addition Ratio (additions / total LoC)")
ax.set_ylabel("Contributors"); ax.set_title("Code Profile Distribution"); ax.legend()

# Role comparison: commits
ax = axes[1]
role_order = ["Author", "Balanced", "Maintainer"]
data = [contrib_df[contrib_df["role"] == r]["commits"].clip(upper=2000).values
        for r in role_order]
bp = ax.boxplot(data, tick_labels=role_order, patch_artist=True, showfliers=False)
colors_box = ["#55A868", "#4C72B0", "#C44E52"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_ylabel("Commits"); ax.set_title("Commit Volume by Role")

# Role comparison: tenure
ax = axes[2]
data = [contrib_df[contrib_df["role"] == r]["tenure_weeks"].clip(upper=300).values
        for r in role_order]
bp = ax.boxplot(data, tick_labels=role_order, patch_artist=True, showfliers=False)
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_ylabel("Tenure (weeks)"); ax.set_title("Tenure by Role")

plt.tight_layout()
plt.savefig(OUT_DIR / "authorship_vs_maintenance.png", dpi=150)
plt.close()
print("  Saved: authorship_vs_maintenance.png")


# ── Chart 6: Combined dashboard ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Untapped Metrics Dashboard — Blockchain OSS Centralization",
             fontsize=16, fontweight="bold")

# 1. Gini commits vs Gini LoC coloured by one-timer fraction
ax = axes[0, 0]
sc = ax.scatter(repo_df["gini_commits"], repo_df["gini_loc"],
                c=repo_df["one_timer_fraction"], cmap="RdYlGn_r",
                alpha=0.6, s=25, edgecolors="black", linewidths=0.3)
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
ax.set_xlabel("Gini (commits)"); ax.set_ylabel("Gini (LoC)")
ax.set_title("Commit vs LoC Gini\n(colour = one-timer %)")
fig.colorbar(sc, ax=ax, label="One-timer fraction", shrink=0.8)

# 2. Retention vs one-timer fraction
ax = axes[0, 1]
ax.scatter(repo_df["one_timer_fraction"], repo_df["retention_6m"],
           alpha=0.4, s=20, c="#8172B2")
rho, p = sp_stats.spearmanr(
    repo_df[["one_timer_fraction","retention_6m"]].dropna()["one_timer_fraction"],
    repo_df[["one_timer_fraction","retention_6m"]].dropna()["retention_6m"])
ax.set_xlabel("One-Timer Fraction"); ax.set_ylabel("6-Month Retention")
ax.set_title(f"One-Timers vs Retention (ρ={rho:.2f})")

# 3. Tenure vs LoC Gini by language
ax = axes[0, 2]
top_langs = ["JavaScript", "TypeScript", "Go", "Rust", "Python", "Solidity"]
LANG_COLORS = {
    "JavaScript": "#F7DF1E", "TypeScript": "#3178C6", "Go": "#00ADD8",
    "Rust": "#DEA584", "Python": "#3776AB", "Solidity": "#363636",
}
for lang in top_langs:
    sub = repo_df[repo_df["_language"] == lang]
    ax.scatter(sub["median_tenure_weeks"], sub["gini_loc"], alpha=0.5, s=25,
               c=LANG_COLORS.get(lang), label=lang, edgecolors="black", linewidths=0.3)
ax.set_xlabel("Median Tenure (weeks)"); ax.set_ylabel("Gini (LoC)")
ax.set_title("Tenure vs LoC Concentration by Language")
ax.legend(fontsize=8, ncol=2)

# 4. Active weeks distribution (log)
ax = axes[1, 0]
ax.hist(contrib_df["active_weeks"].clip(upper=200), bins=50,
        edgecolor="black", alpha=0.7, color="#DD8452", log=True)
ax.set_xlabel("Active Weeks per Contributor"); ax.set_ylabel("Count (log)")
ax.set_title(f"Active Weeks Distribution\nMedian={contrib_df['active_weeks'].median():.0f}")

# 5. Role pie chart
ax = axes[1, 1]
role_counts = contrib_df["role"].value_counts()
colors_pie = ["#55A868", "#4C72B0", "#C44E52"]
ax.pie(role_counts.values, labels=role_counts.index, colors=colors_pie,
       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
ax.set_title("Contributor Role Distribution\n(by addition ratio)")

# 6. Stars vs one-timer fraction
ax = axes[1, 2]
ax.scatter(repo_df["_stars"], repo_df["one_timer_fraction"],
           alpha=0.4, s=20, c="#4C72B0")
ax.set_xscale("log")
rho, p = sp_stats.spearmanr(
    repo_df[["_stars","one_timer_fraction"]].dropna()["_stars"],
    repo_df[["_stars","one_timer_fraction"]].dropna()["one_timer_fraction"])
ax.set_xlabel("Stars (log)"); ax.set_ylabel("One-Timer Fraction")
ax.set_title(f"Popularity vs One-Timer Rate (ρ={rho:.2f})")

plt.tight_layout()
plt.savefig(OUT_DIR / "untapped_dashboard.png", dpi=150)
plt.close()
print("  Saved: untapped_dashboard.png")


# ==============================================================
print("\n" + "=" * 65)
print("  UNTAPPED ANALYSIS COMPLETE")
print("=" * 65)
print(f"\n  Output files:")
for f in sorted(OUT_DIR.glob("untapped*")):
    print(f"    {f.name:<50s} {f.stat().st_size/1024:.1f} KB")
for f in sorted(OUT_DIR.glob("loc_*")) | sorted(OUT_DIR.glob("tenure*")) | \
         sorted(OUT_DIR.glob("arrival*")) | sorted(OUT_DIR.glob("consist*")) | \
         sorted(OUT_DIR.glob("authorship*")):
    print(f"    {f.name:<50s} {f.stat().st_size/1024:.1f} KB")
