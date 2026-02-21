#!/usr/bin/env python3
"""
Extended Centralization Analyses
=================================
Runs additional analyses on top of the base centralization_analysis.py results:

  1. Language vs centralization
  2. Repo age vs centralization (+ regression)
  3. Shannon entropy as a complementary concentration measure
  4. Contributor overlap / cross-pollination network

All data comes from existing CSVs and cached API responses — no new API calls.
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
OUT_DIR     = SCRIPT_DIR / "output"
CACHE_DIR   = SCRIPT_DIR / "cache"

SUMMARY_CSV = OUT_DIR / "centralization_summary.csv"
SAMPLE_CSV  = OUT_DIR / "sampled_repos.csv"

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
summary = pd.read_csv(SUMMARY_CSV)
sample  = pd.read_csv(SAMPLE_CSV)

# Merge sample metadata into summary
df = sample[["owner_repo", "_language", "_age_weeks", "_created_at", "_stars",
             "category", "subcategory", "star_bin"]].merge(
    summary, on="owner_repo", how="inner"
)
print(f"  Merged dataset: {len(df)} repos")


# ==============================================================
# ANALYSIS 1: Language vs Centralization
# ==============================================================
print("\n" + "=" * 65)
print("  ANALYSIS 1: Language vs Centralization")
print("=" * 65)

# Keep languages with >= 5 repos for statistical stability
lang_counts = df["_language"].value_counts()
major_langs = lang_counts[lang_counts >= 5].index.tolist()
df_lang = df[df["_language"].isin(major_langs)].copy()

print(f"\n  Languages with >= 5 repos: {len(major_langs)}")

# Per-language statistics
lang_stats = df_lang.groupby("_language").agg(
    n=("gini_commits", "count"),
    med_gini=("gini_commits", "median"),
    mean_gini=("gini_commits", "mean"),
    med_hhi=("hhi_commits", "median"),
    mean_hhi=("hhi_commits", "mean"),
    med_bus=("bus_factor_80pct", "median"),
    med_authors=("n_authors", "median"),
    med_stars=("_stars", "median"),
).sort_values("n", ascending=False)

print("\n  Language Breakdown:")
print(f"  {'Language':<16s} {'n':>4s} {'Med Gini':>9s} {'Med HHI':>9s} "
      f"{'Med BF':>7s} {'Med Auth':>9s} {'Med Stars':>10s}")
print(f"  {'-'*64}")
for lang, row in lang_stats.iterrows():
    print(f"  {lang:<16s} {row.n:>4.0f} {row.med_gini:>9.3f} {row.med_hhi:>9.3f} "
          f"{row.med_bus:>7.0f} {row.med_authors:>9.0f} {row.med_stars:>10.0f}")

lang_stats.to_csv(OUT_DIR / "language_vs_centralization.csv")

# Kruskal-Wallis test across major languages
groups_gini = [g["gini_commits"].dropna().values for _, g in df_lang.groupby("_language")]
groups_hhi  = [g["hhi_commits"].dropna().values for _, g in df_lang.groupby("_language")]
groups_bf   = [g["bus_factor_80pct"].dropna().values for _, g in df_lang.groupby("_language")]

kw_gini = sp_stats.kruskal(*groups_gini)
kw_hhi  = sp_stats.kruskal(*groups_hhi)
kw_bf   = sp_stats.kruskal(*groups_bf)

print(f"\n  Kruskal-Wallis test (language differences):")
print(f"    Gini:       H={kw_gini.statistic:.2f}, p={kw_gini.pvalue:.4f} "
      f"{'***' if kw_gini.pvalue < 0.001 else '**' if kw_gini.pvalue < 0.01 else '*' if kw_gini.pvalue < 0.05 else 'ns'}")
print(f"    HHI:        H={kw_hhi.statistic:.2f}, p={kw_hhi.pvalue:.4f} "
      f"{'***' if kw_hhi.pvalue < 0.001 else '**' if kw_hhi.pvalue < 0.01 else '*' if kw_hhi.pvalue < 0.05 else 'ns'}")
print(f"    Bus Factor: H={kw_bf.statistic:.2f}, p={kw_bf.pvalue:.4f} "
      f"{'***' if kw_bf.pvalue < 0.001 else '**' if kw_bf.pvalue < 0.01 else '*' if kw_bf.pvalue < 0.05 else 'ns'}")

# Pairwise Mann-Whitney for top languages vs rest
print(f"\n  Pairwise Mann-Whitney (language vs all others):")
top_langs = lang_stats.head(8).index.tolist()
pairwise_results = []
for lang in top_langs:
    in_lang = df_lang[df_lang["_language"] == lang]["gini_commits"].dropna()
    not_lang = df_lang[df_lang["_language"] != lang]["gini_commits"].dropna()
    if len(in_lang) >= 3:
        u, p = sp_stats.mannwhitneyu(in_lang, not_lang, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        pairwise_results.append({"language": lang, "n": len(in_lang),
                                  "med_gini": in_lang.median(), "U": u, "p": p, "sig": sig})
        print(f"    {lang:<16s} (n={len(in_lang):>3d}) med_gini={in_lang.median():.3f}  "
              f"U={u:.0f}, p={p:.4f} {sig}")

# Plot: boxplot by language
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Centralization Metrics by Programming Language", fontsize=14, fontweight="bold")

for ax, metric, title in zip(axes,
    ["gini_commits", "hhi_commits", "bus_factor_80pct"],
    ["Gini Coefficient", "HHI", "Bus Factor (80%)"]):
    order = lang_stats.head(10).index.tolist()
    data = [df_lang[df_lang["_language"] == l][metric].dropna().values for l in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, vert=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(plt.cm.tab10(i / 10)); patch.set_alpha(0.6)
    ax.set_title(title); ax.set_ylabel(title)
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUT_DIR / "language_vs_centralization.png", dpi=150)
plt.close()
print(f"\n  Saved: language_vs_centralization.png")


# ==============================================================
# ANALYSIS 2: Repo Age vs Centralization
# ==============================================================
print("\n" + "=" * 65)
print("  ANALYSIS 2: Repo Age vs Centralization")
print("=" * 65)

df["age_years"] = df["_age_weeks"] / 52.0

# Spearman rank correlation (robust to non-linearity)
for metric in ["gini_commits", "hhi_commits", "bus_factor_80pct", "n_authors"]:
    valid = df[["age_years", metric]].dropna()
    rho, p = sp_stats.spearmanr(valid["age_years"], valid[metric])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  Spearman(age, {metric:<20s}): rho={rho:+.3f}, p={p:.4f} {sig}")

# Age bins (quartiles)
df["age_bin"] = pd.qcut(df["age_years"], q=4, labels=["<Q1 (youngest)", "Q1-Q2", "Q2-Q3", "Q4 (oldest)"])
age_stats = df.groupby("age_bin", observed=True).agg(
    n=("gini_commits", "count"),
    med_gini=("gini_commits", "median"),
    med_hhi=("hhi_commits", "median"),
    med_bf=("bus_factor_80pct", "median"),
    med_authors=("n_authors", "median"),
    med_commits=("total_commits", "median"),
    age_range_years=("age_years", lambda x: f"{x.min():.1f}–{x.max():.1f}"),
)
print(f"\n  Age quartile breakdown:")
print(age_stats.to_string())
age_stats.to_csv(OUT_DIR / "age_vs_centralization.csv")

# Linear regression: age_years vs Gini
from numpy.polynomial import polynomial as P
valid = df[["age_years", "gini_commits"]].dropna()
slope, intercept, r_value, p_value, std_err = sp_stats.linregress(valid["age_years"], valid["gini_commits"])
print(f"\n  OLS Regression: Gini = {intercept:.3f} + {slope:.4f} × age_years")
print(f"    R² = {r_value**2:.4f}, p = {p_value:.4f}, SE(slope) = {std_err:.4f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Repository Age vs Centralization", fontsize=14, fontweight="bold")

# Scatter + regression for Gini
ax = axes[0]
ax.scatter(df["age_years"], df["gini_commits"], alpha=0.4, s=15, color="#4C72B0")
xs = np.linspace(0, df["age_years"].max(), 100)
ax.plot(xs, intercept + slope * xs, "r--", lw=2,
        label=f"OLS: Gini={intercept:.2f}+{slope:.3f}×age\nR²={r_value**2:.3f}, p={p_value:.3f}")
ax.set_xlabel("Repo Age (years)"); ax.set_ylabel("Gini"); ax.set_title("Gini vs Age")
ax.legend(fontsize=9)

# Scatter for HHI
ax = axes[1]
slope2, intercept2, r2, p2, se2 = sp_stats.linregress(
    df[["age_years","hhi_commits"]].dropna()["age_years"],
    df[["age_years","hhi_commits"]].dropna()["hhi_commits"])
ax.scatter(df["age_years"], df["hhi_commits"], alpha=0.4, s=15, color="#55A868")
ax.plot(xs, intercept2 + slope2 * xs, "r--", lw=2,
        label=f"OLS: HHI={intercept2:.2f}+{slope2:.3f}×age\nR²={r2**2:.3f}, p={p2:.3f}")
ax.set_xlabel("Repo Age (years)"); ax.set_ylabel("HHI"); ax.set_title("HHI vs Age")
ax.legend(fontsize=9)

# Scatter for bus factor
ax = axes[2]
slope3, intercept3, r3, p3, se3 = sp_stats.linregress(
    df[["age_years","bus_factor_80pct"]].dropna()["age_years"],
    df[["age_years","bus_factor_80pct"]].dropna()["bus_factor_80pct"])
ax.scatter(df["age_years"], df["bus_factor_80pct"], alpha=0.4, s=15, color="#C44E52")
ax.plot(xs, intercept3 + slope3 * xs, "r--", lw=2,
        label=f"OLS: BF={intercept3:.1f}+{slope3:.2f}×age\nR²={r3**2:.3f}, p={p3:.3f}")
ax.set_xlabel("Repo Age (years)"); ax.set_ylabel("Bus Factor"); ax.set_title("Bus Factor vs Age")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "age_vs_centralization.png", dpi=150)
plt.close()
print(f"  Saved: age_vs_centralization.png")


# ==============================================================
# ANALYSIS 3: Shannon Entropy
# ==============================================================
print("\n" + "=" * 65)
print("  ANALYSIS 3: Shannon Entropy")
print("=" * 65)

def shannon_entropy(counts):
    """Shannon entropy in bits from raw counts."""
    arr = np.array(counts, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    total = arr.sum()
    p = arr / total
    return float(-np.sum(p * np.log2(p)))

def normalized_entropy(counts):
    """Shannon entropy normalized to [0,1] (H / log2(n))."""
    arr = np.array(counts, dtype=float)
    arr = arr[arr > 0]
    n = arr.size
    if n <= 1:
        return 0.0
    h = shannon_entropy(arr)
    return h / np.log2(n)

# Compute entropy for each repo from cached stats
entropy_rows = []
for _, row in summary.iterrows():
    owner_repo = row["owner_repo"]
    cache_file = CACHE_DIR / f"{owner_repo.replace('/', '__')}.stats_contributors.json"
    if not cache_file.exists():
        continue
    with open(cache_file) as f:
        stats = json.load(f)
    counts = [a.get("total", 0) for a in stats if a.get("total", 0) > 0]
    if not counts:
        continue
    entropy_rows.append({
        "owner_repo": owner_repo,
        "shannon_entropy": shannon_entropy(counts),
        "normalized_entropy": normalized_entropy(counts),
        "max_entropy": np.log2(len(counts)) if len(counts) > 1 else 0.0,
        "n_active_authors": len(counts),
    })

entropy_df = pd.DataFrame(entropy_rows)
entropy_merged = df.merge(entropy_df, on="owner_repo", how="inner")

print(f"  Computed entropy for {len(entropy_df)} repos")
print(f"\n  Shannon Entropy Statistics:")
print(f"    {'Metric':<25s} {'Median':>9s} {'Mean':>9s} {'Std':>9s}")
print(f"    {'-'*52}")
for col in ["shannon_entropy", "normalized_entropy"]:
    vals = entropy_merged[col]
    print(f"    {col:<25s} {vals.median():>9.3f} {vals.mean():>9.3f} {vals.std():>9.3f}")

# Correlations between entropy and other metrics
print(f"\n  Correlations with normalized entropy:")
for metric in ["gini_commits", "hhi_commits", "top1_share", "bus_factor_80pct", "n_authors", "_stars"]:
    valid = entropy_merged[[metric, "normalized_entropy"]].dropna()
    rho, p = sp_stats.spearmanr(valid[metric], valid["normalized_entropy"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    Spearman(norm_entropy, {metric:<20s}): rho={rho:+.3f}, p={p:.6f} {sig}")

# Entropy by language
ent_by_lang = entropy_merged[entropy_merged["_language"].isin(major_langs)].groupby("_language").agg(
    n=("normalized_entropy", "count"),
    med_norm_entropy=("normalized_entropy", "median"),
    mean_norm_entropy=("normalized_entropy", "mean"),
).sort_values("n", ascending=False)
print(f"\n  Normalized Entropy by Language:")
print(ent_by_lang.head(10).to_string())

entropy_merged[["owner_repo", "shannon_entropy", "normalized_entropy",
                "max_entropy", "n_active_authors"]].to_csv(
    OUT_DIR / "shannon_entropy.csv", index=False)

# Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Shannon Entropy Analysis", fontsize=14, fontweight="bold")

ax = axes[0]
ax.hist(entropy_merged["normalized_entropy"].dropna(), bins=25, edgecolor="black", alpha=0.7, color="#4C72B0")
med = entropy_merged["normalized_entropy"].median()
ax.axvline(med, color="red", ls="--", label=f"Median = {med:.2f}")
ax.set_xlabel("Normalized Entropy"); ax.set_ylabel("Repos"); ax.set_title("Distribution of Normalized Entropy")
ax.legend()

ax = axes[1]
ax.scatter(entropy_merged["gini_commits"], entropy_merged["normalized_entropy"], alpha=0.4, s=15, color="#55A868")
rho, _ = sp_stats.spearmanr(entropy_merged["gini_commits"].dropna(), entropy_merged["normalized_entropy"].dropna())
ax.set_xlabel("Gini"); ax.set_ylabel("Normalized Entropy")
ax.set_title(f"Gini vs Norm. Entropy (ρ={rho:.2f})")

ax = axes[2]
ax.scatter(entropy_merged["hhi_commits"], entropy_merged["normalized_entropy"], alpha=0.4, s=15, color="#C44E52")
rho, _ = sp_stats.spearmanr(entropy_merged["hhi_commits"].dropna(), entropy_merged["normalized_entropy"].dropna())
ax.set_xlabel("HHI"); ax.set_ylabel("Normalized Entropy")
ax.set_title(f"HHI vs Norm. Entropy (ρ={rho:.2f})")

plt.tight_layout()
plt.savefig(OUT_DIR / "shannon_entropy.png", dpi=150)
plt.close()
print(f"  Saved: shannon_entropy.png")


# ==============================================================
# ANALYSIS 5: Contributor Overlap / Cross-Pollination
# ==============================================================
print("\n" + "=" * 65)
print("  ANALYSIS 5: Contributor Overlap / Cross-Pollination")
print("=" * 65)

# Build developer → repos mapping from cached stats
dev_repos = defaultdict(set)       # login → set of repos
repo_devs = defaultdict(set)       # repo → set of logins
repo_category = dict(zip(sample["owner_repo"], sample["category"]))

for cache_file in sorted(CACHE_DIR.glob("*.stats_contributors.json")):
    repo_name = cache_file.stem.replace(".stats_contributors", "").replace("__", "/")
    with open(cache_file) as f:
        stats = json.load(f)
    for a in stats:
        login = (a.get("author") or {}).get("login")
        if not login or login.startswith("id:"):
            continue
        total = a.get("total", 0)
        if total > 0:  # only count active contributors
            dev_repos[login].add(repo_name)
            repo_devs[repo_name].add(login)

total_devs = len(dev_repos)
total_repos_with_devs = len(repo_devs)

# Multi-repo contributors
multi_repo_devs = {d: repos for d, repos in dev_repos.items() if len(repos) >= 2}
multi3_devs = {d: repos for d, repos in dev_repos.items() if len(repos) >= 3}
multi5_devs = {d: repos for d, repos in dev_repos.items() if len(repos) >= 5}

print(f"\n  Total unique developers:       {total_devs:,}")
print(f"  Developers in >= 2 repos:      {len(multi_repo_devs):,} ({100*len(multi_repo_devs)/total_devs:.1f}%)")
print(f"  Developers in >= 3 repos:      {len(multi3_devs):,} ({100*len(multi3_devs)/total_devs:.1f}%)")
print(f"  Developers in >= 5 repos:      {len(multi5_devs):,} ({100*len(multi5_devs)/total_devs:.1f}%)")

# Distribution of repos per developer
repos_per_dev = [len(r) for r in dev_repos.values()]
print(f"\n  Repos per developer:")
print(f"    Median: {np.median(repos_per_dev):.0f}")
print(f"    Mean:   {np.mean(repos_per_dev):.2f}")
print(f"    Max:    {max(repos_per_dev)}")
print(f"    P90:    {np.percentile(repos_per_dev, 90):.0f}")
print(f"    P99:    {np.percentile(repos_per_dev, 99):.0f}")

# Top cross-pollinating developers
top_cross = sorted(multi_repo_devs.items(), key=lambda x: len(x[1]), reverse=True)[:20]
print(f"\n  Top 20 cross-pollinating developers:")
print(f"  {'Developer':<30s} {'Repos':>6s} {'Ecosystems':>11s}")
print(f"  {'-'*47}")
cross_poll_rows = []
for login, repos in top_cross:
    ecosystems = set()
    for r in repos:
        cat = repo_category.get(r, "unknown")
        if pd.notna(cat):
            ecosystems.add(cat)
    cross_poll_rows.append({
        "developer": login, "n_repos": len(repos),
        "n_ecosystems": len(ecosystems),
        "ecosystems": ", ".join(sorted(ecosystems)),
        "repos": ", ".join(sorted(repos)),
    })
    print(f"  {login:<30s} {len(repos):>6d} {len(ecosystems):>11d}")

pd.DataFrame(cross_poll_rows).to_csv(OUT_DIR / "top_cross_pollinators.csv", index=False)

# Cross-ecosystem overlap: how many devs contribute to 2+ ecosystems?
dev_ecosystems = {}
for login, repos in dev_repos.items():
    cats = set()
    for r in repos:
        cat = repo_category.get(r, None)
        if cat and pd.notna(cat):
            cats.add(cat)
    dev_ecosystems[login] = cats

multi_eco_devs = {d: e for d, e in dev_ecosystems.items() if len(e) >= 2}
print(f"\n  Cross-ecosystem developers:")
print(f"    Devs contributing to 1 ecosystem:  {sum(1 for e in dev_ecosystems.values() if len(e) == 1):,}")
print(f"    Devs contributing to 2+ ecosystems: {len(multi_eco_devs):,} "
      f"({100*len(multi_eco_devs)/total_devs:.1f}%)")

# Ecosystem co-occurrence matrix
ecosystems_list = sorted(set(c for cats in dev_ecosystems.values() for c in cats if c))
n_eco = len(ecosystems_list)
eco_idx = {e: i for i, e in enumerate(ecosystems_list)}
cooccurrence = np.zeros((n_eco, n_eco), dtype=int)
for login, cats in dev_ecosystems.items():
    cats_list = sorted(cats)
    for i, a in enumerate(cats_list):
        for b in cats_list[i:]:
            cooccurrence[eco_idx[a], eco_idx[b]] += 1
            if a != b:
                cooccurrence[eco_idx[b], eco_idx[a]] += 1

# Top ecosystem pairs by shared developers
print(f"\n  Top ecosystem pairs by shared developers:")
eco_pairs = []
for i in range(n_eco):
    for j in range(i + 1, n_eco):
        if cooccurrence[i, j] > 0:
            eco_pairs.append((ecosystems_list[i], ecosystems_list[j], cooccurrence[i, j]))
eco_pairs.sort(key=lambda x: x[2], reverse=True)
print(f"  {'Ecosystem A':<20s} {'Ecosystem B':<20s} {'Shared Devs':>12s}")
print(f"  {'-'*52}")
for a, b, count in eco_pairs[:15]:
    print(f"  {a:<20s} {b:<20s} {count:>12d}")

pd.DataFrame(eco_pairs, columns=["ecosystem_a", "ecosystem_b", "shared_developers"]).to_csv(
    OUT_DIR / "ecosystem_overlap.csv", index=False)

# Repo connectivity: how many repo pairs share at least 1 developer?
print(f"\n  Repo connectivity:")
repo_list = sorted(repo_devs.keys())
n_repos_graph = len(repo_list)
shared_pairs = 0
repos_with_shared = set()
for i in range(n_repos_graph):
    for j in range(i + 1, n_repos_graph):
        overlap = repo_devs[repo_list[i]] & repo_devs[repo_list[j]]
        if overlap:
            shared_pairs += 1
            repos_with_shared.add(repo_list[i])
            repos_with_shared.add(repo_list[j])

max_pairs = n_repos_graph * (n_repos_graph - 1) // 2
print(f"    Total repo pairs:          {max_pairs:,}")
print(f"    Pairs sharing >= 1 dev:    {shared_pairs:,} ({100*shared_pairs/max_pairs:.1f}%)")
print(f"    Repos with >= 1 shared dev: {len(repos_with_shared)} / {n_repos_graph} "
      f"({100*len(repos_with_shared)/n_repos_graph:.1f}%)")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Contributor Cross-Pollination Analysis", fontsize=14, fontweight="bold")

# Repos per developer distribution
ax = axes[0]
bins = list(range(1, min(max(repos_per_dev) + 2, 20))) + [max(repos_per_dev) + 1]
ax.hist(repos_per_dev, bins=bins, edgecolor="black", alpha=0.7, color="#4C72B0", log=True)
ax.set_xlabel("Repos per Developer"); ax.set_ylabel("Count (log scale)")
ax.set_title(f"Developer Participation\n({len(multi_repo_devs):,}/{total_devs:,} in 2+ repos)")

# Ecosystem co-occurrence heatmap (top 10 ecosystems by size)
ax = axes[1]
top_ecos = [e for e in ecosystems_list if e in eco_idx]
eco_sizes = {e: cooccurrence[eco_idx[e], eco_idx[e]] for e in top_ecos}
top_ecos = sorted(eco_sizes, key=eco_sizes.get, reverse=True)[:12]
top_idx = [eco_idx[e] for e in top_ecos]
sub_matrix = cooccurrence[np.ix_(top_idx, top_idx)]
im = ax.imshow(sub_matrix, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(top_ecos)))
ax.set_xticklabels(top_ecos, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(top_ecos)))
ax.set_yticklabels(top_ecos, fontsize=8)
ax.set_title("Developer Overlap Between Ecosystems")
for i in range(len(top_ecos)):
    for j in range(len(top_ecos)):
        val = sub_matrix[i, j]
        if val > 0:
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7,
                    color="white" if val > sub_matrix.max() * 0.6 else "black")
fig.colorbar(im, ax=ax, label="Shared Developers", shrink=0.8)

# Top cross-pollinators bar chart
ax = axes[2]
top_n = min(15, len(top_cross))
names = [x[0] for x in top_cross[:top_n]]
counts = [len(x[1]) for x in top_cross[:top_n]]
bars = ax.barh(range(top_n), counts, color="#55A868", edgecolor="black", alpha=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Number of Repos")
ax.set_title("Top Cross-Pollinating Developers")

plt.tight_layout()
plt.savefig(OUT_DIR / "cross_pollination.png", dpi=150)
plt.close()
print(f"  Saved: cross_pollination.png")


# ==============================================================
# SUMMARY
# ==============================================================
print("\n" + "=" * 65)
print("  EXTENDED ANALYSIS COMPLETE")
print("=" * 65)
print(f"\n  Output files:")
for f in sorted(OUT_DIR.glob("*")):
    if f.suffix in (".csv", ".png"):
        print(f"    {f.name:<45s} {f.stat().st_size/1024:.1f} KB")
