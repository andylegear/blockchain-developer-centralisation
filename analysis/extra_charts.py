#!/usr/bin/env python3
"""
Extra visualisations from the extended analysis results.
Generates charts that combine multiple dimensions for richer insight.
"""

import json, math
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "output"
CACHE_DIR  = SCRIPT_DIR / "cache"

# ── Load all datasets ────────────────────────────────────────
summary  = pd.read_csv(OUT_DIR / "centralization_summary.csv")
sample   = pd.read_csv(OUT_DIR / "sampled_repos.csv")
entropy  = pd.read_csv(OUT_DIR / "shannon_entropy.csv")

df = sample[["owner_repo", "_language", "_age_weeks", "_stars",
             "category", "star_bin"]].merge(
    summary, on="owner_repo", how="inner"
).merge(entropy, on="owner_repo", how="inner")

df["age_years"] = df["_age_weeks"] / 52.0

# Languages with enough repos
lang_counts = df["_language"].value_counts()
major_langs = lang_counts[lang_counts >= 5].index.tolist()
TOP_LANGS   = lang_counts.head(8).index.tolist()

# Colour palette for languages
LANG_COLORS = {
    "JavaScript": "#F7DF1E", "TypeScript": "#3178C6", "Go": "#00ADD8",
    "Rust": "#DEA584", "Python": "#3776AB", "Solidity": "#363636",
    "C++": "#00599C", "HTML": "#E34F26", "Java": "#B07219",
    "C#": "#68217A", "Shell": "#89E051",
}

# ================================================================
# CHART 1 — Radar / Spider Chart: Language Centralization Profiles
# ================================================================
print("Chart 1: Radar chart – language profiles ...")

metrics_radar = ["gini_commits", "hhi_commits", "normalized_entropy",
                 "bus_factor_80pct", "n_authors"]
labels_radar  = ["Gini", "HHI", "Norm. Entropy", "Bus Factor", "Authors"]

# Normalise each metric to [0,1] across the full dataset for fair comparison
norm_vals = {}
for m in metrics_radar:
    vals = df[m].dropna()
    norm_vals[m] = (vals.min(), vals.max())

def normalise(val, metric):
    lo, hi = norm_vals[metric]
    return (val - lo) / (hi - lo) if hi > lo else 0.5

radar_data = {}
for lang in TOP_LANGS:
    sub = df[df["_language"] == lang]
    radar_data[lang] = [normalise(sub[m].median(), m) for m in metrics_radar]

angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.suptitle("Language Centralization Profiles (Median, Normalised)",
             fontsize=14, fontweight="bold", y=0.98)

for lang, vals in radar_data.items():
    vals_closed = vals + vals[:1]
    color = LANG_COLORS.get(lang, "#999999")
    ax.plot(angles, vals_closed, "o-", linewidth=2, label=lang, color=color)
    ax.fill(angles, vals_closed, alpha=0.08, color=color)

ax.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75])
ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=8, color="grey")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "radar_language_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: radar_language_profiles.png")


# ================================================================
# CHART 2 — Gini vs Normalised Entropy scatter, coloured by language
# ================================================================
print("Chart 2: Gini vs Entropy scatter ...")

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Gini vs Normalised Shannon Entropy by Language",
             fontsize=14, fontweight="bold")

# Background: all repos in light grey
others = df[~df["_language"].isin(TOP_LANGS)]
ax.scatter(others["gini_commits"], others["normalized_entropy"],
           alpha=0.25, s=15, c="lightgrey", label="_nolegend_")

# Overlay top languages
handles = []
for lang in TOP_LANGS:
    sub = df[df["_language"] == lang]
    color = LANG_COLORS.get(lang, "#999999")
    ax.scatter(sub["gini_commits"], sub["normalized_entropy"],
               alpha=0.55, s=30, c=color, edgecolors="black", linewidths=0.3)
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                          markersize=8, label=f"{lang} (n={len(sub)})"))

# Trend line
valid = df[["gini_commits", "normalized_entropy"]].dropna()
slope, intercept, r, p, se = sp_stats.linregress(valid["gini_commits"],
                                                   valid["normalized_entropy"])
xs = np.linspace(0, 1, 100)
ax.plot(xs, intercept + slope * xs, "r--", lw=1.5, alpha=0.7,
        label=f"OLS: ρ={r:.2f}, p={p:.4f}")
handles.append(Line2D([0], [0], color="red", ls="--", label=f"OLS (r={r:.2f})"))

ax.set_xlabel("Gini Coefficient", fontsize=12)
ax.set_ylabel("Normalised Shannon Entropy", fontsize=12)
ax.legend(handles=handles, fontsize=9, loc="lower left")
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(OUT_DIR / "gini_vs_entropy_by_language.png", dpi=150)
plt.close()
print("  Saved: gini_vs_entropy_by_language.png")


# ================================================================
# CHART 3 — Bubble chart: Age vs Gini, size = stars, colour = lang
# ================================================================
print("Chart 3: Bubble chart – age vs Gini ...")

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle("Repository Age vs Gini — Bubble Size = Stars, Colour = Language",
             fontsize=14, fontweight="bold")

# Bubble size (log-scaled so outliers don't dominate)
sizes = np.clip(np.log1p(df["_stars"]) * 12, 8, 200)

for lang in TOP_LANGS:
    mask = df["_language"] == lang
    ax.scatter(df.loc[mask, "age_years"], df.loc[mask, "gini_commits"],
               s=sizes[mask], alpha=0.5, label=lang,
               c=LANG_COLORS.get(lang, "#999"), edgecolors="black", linewidths=0.3)

# Non-top languages
mask_other = ~df["_language"].isin(TOP_LANGS)
ax.scatter(df.loc[mask_other, "age_years"], df.loc[mask_other, "gini_commits"],
           s=sizes[mask_other], alpha=0.2, c="lightgrey", edgecolors="grey",
           linewidths=0.2, label="Other")

ax.set_xlabel("Repository Age (years)", fontsize=12)
ax.set_ylabel("Gini Coefficient", fontsize=12)
ax.legend(fontsize=9, loc="lower right", ncol=3)
ax.axhline(df["gini_commits"].median(), color="red", ls=":", alpha=0.5,
           label="Median Gini")

# Size legend
for stars, label in [(10, "10★"), (100, "100★"), (1000, "1k★"), (10000, "10k★")]:
    s = np.log1p(stars) * 12
    ax.scatter([], [], s=s, c="grey", alpha=0.5, edgecolors="black",
               linewidths=0.3, label=label)
ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.8)
plt.tight_layout()
plt.savefig(OUT_DIR / "bubble_age_gini_stars.png", dpi=150)
plt.close()
print("  Saved: bubble_age_gini_stars.png")


# ================================================================
# CHART 4 — Ecosystem Network Graph (shared developers)
# ================================================================
print("Chart 4: Ecosystem network graph ...")

eco_overlap = pd.read_csv(OUT_DIR / "ecosystem_overlap.csv")

# Only edges with >= 3 shared devs for readability
edges = eco_overlap[eco_overlap["shared_developers"] >= 3].copy()

# Node set
nodes = sorted(set(edges["ecosystem_a"]) | set(edges["ecosystem_b"]))
n = len(nodes)

# Compute node sizes from developer counts
dev_repos = defaultdict(set)
repo_category = dict(zip(sample["owner_repo"], sample["category"]))
for cache_file in CACHE_DIR.glob("*.stats_contributors.json"):
    repo_name = cache_file.stem.replace(".stats_contributors", "").replace("__", "/")
    cat = repo_category.get(repo_name)
    if not cat or cat not in nodes:
        continue
    with open(cache_file) as f:
        stats = json.load(f)
    for a in stats:
        login = (a.get("author") or {}).get("login")
        if login and a.get("total", 0) > 0:
            dev_repos[login].add(cat)

eco_dev_count = defaultdict(int)
for login, cats in dev_repos.items():
    for c in cats:
        eco_dev_count[c] += 1

# Circular layout
node_idx = {nd: i for i, nd in enumerate(nodes)}
theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos_x = np.cos(theta)
pos_y = np.sin(theta)

fig, ax = plt.subplots(figsize=(12, 12))
fig.suptitle("Ecosystem Developer Overlap Network\n(edges = shared developers, "
             "node size = total devs in ecosystem)",
             fontsize=14, fontweight="bold")

# Draw edges
max_weight = edges["shared_developers"].max()
for _, row in edges.iterrows():
    i, j = node_idx[row["ecosystem_a"]], node_idx[row["ecosystem_b"]]
    w = row["shared_developers"]
    lw = 0.5 + 5 * (w / max_weight)
    alpha = 0.2 + 0.6 * (w / max_weight)
    ax.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]],
            color="#4C72B0", lw=lw, alpha=alpha, zorder=1)
    # Label the edge weight at midpoint
    mx, my = (pos_x[i] + pos_x[j]) / 2, (pos_y[i] + pos_y[j]) / 2
    ax.text(mx, my, str(w), fontsize=7, ha="center", va="center",
            color="#333", alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

# Draw nodes
for nd in nodes:
    i = node_idx[nd]
    size = max(eco_dev_count.get(nd, 5), 5)
    s = 50 + 15 * math.sqrt(size)
    ax.scatter(pos_x[i], pos_y[i], s=s**1.1, c="#DD8452", edgecolors="black",
               linewidths=1.2, zorder=2, alpha=0.85)
    # Label
    offset = 0.12
    lx = pos_x[i] * (1 + offset)
    ly = pos_y[i] * (1 + offset)
    ha = "left" if pos_x[i] >= 0 else "right"
    angle = math.degrees(theta[i])
    if 90 < angle < 270:
        angle -= 180
    ax.text(lx, ly, nd, fontsize=9, fontweight="bold", ha=ha, va="center",
            rotation=angle, rotation_mode="anchor")

ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "ecosystem_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ecosystem_network.png")


# ================================================================
# CHART 5 — Violin plots: Gini & Entropy by language
# ================================================================
print("Chart 5: Violin plots ...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Distribution of Centralization Metrics by Language",
             fontsize=14, fontweight="bold")

for ax, metric, title in zip(axes,
    ["gini_commits", "normalized_entropy"],
    ["Gini Coefficient", "Normalised Shannon Entropy"]):

    data = [df[df["_language"] == l][metric].dropna().values for l in TOP_LANGS]
    parts = ax.violinplot(data, positions=range(len(TOP_LANGS)),
                          showmeans=True, showmedians=True, showextrema=False)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(LANG_COLORS.get(TOP_LANGS[i], "#999"))
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("red")
    parts["cmedians"].set_color("black")

    # Overlay individual points (jittered)
    for i, lang in enumerate(TOP_LANGS):
        vals = df[df["_language"] == lang][metric].dropna().values
        jitter = np.random.default_rng(42).normal(0, 0.06, len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals, s=8, alpha=0.4,
                   c=LANG_COLORS.get(lang, "#999"), edgecolors="none")

    ax.set_xticks(range(len(TOP_LANGS)))
    ax.set_xticklabels(TOP_LANGS, rotation=45, ha="right", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(title)

    # Custom legend
    ax.legend(handles=[
        Line2D([0], [0], color="black", label="Median"),
        Line2D([0], [0], color="red", label="Mean"),
    ], fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "violin_language_metrics.png", dpi=150)
plt.close()
print("  Saved: violin_language_metrics.png")


# ================================================================
# CHART 6 — CDF of developer participation + Lorenz curve
# ================================================================
print("Chart 6: Developer participation CDF + Lorenz curve ...")

# Build developer commit totals
dev_commits = defaultdict(int)
dev_repos_map = defaultdict(set)
for cache_file in CACHE_DIR.glob("*.stats_contributors.json"):
    repo_name = cache_file.stem.replace(".stats_contributors", "").replace("__", "/")
    with open(cache_file) as f:
        stats = json.load(f)
    for a in stats:
        login = (a.get("author") or {}).get("login")
        if login and a.get("total", 0) > 0:
            dev_commits[login] += a["total"]
            dev_repos_map[login].add(repo_name)

repos_per_dev = np.array([len(r) for r in dev_repos_map.values()])
commits_per_dev = np.array(list(dev_commits.values()))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Developer Participation Structure", fontsize=14, fontweight="bold")

# Panel A: CDF of repos per developer
ax = axes[0]
sorted_rpd = np.sort(repos_per_dev)
cdf_y = np.arange(1, len(sorted_rpd) + 1) / len(sorted_rpd)
ax.step(sorted_rpd, cdf_y, where="post", color="#4C72B0", lw=2)
# Mark key thresholds
for thresh, label, color in [(1, "1 repo", "#55A868"), (2, "2 repos", "#C44E52"),
                               (5, "5 repos", "#8172B2")]:
    frac = np.mean(repos_per_dev <= thresh)
    ax.axvline(thresh, color=color, ls="--", alpha=0.5)
    ax.text(thresh + 0.3, frac - 0.05, f"{frac:.1%} ≤ {label}",
            fontsize=9, color=color)
ax.set_xlabel("Repos per Developer"); ax.set_ylabel("Cumulative Fraction")
ax.set_title("CDF: Repos per Developer")
ax.set_xlim(0, 20); ax.set_ylim(0, 1.02)

# Panel B: Lorenz curve for commits
ax = axes[1]
sorted_c = np.sort(commits_per_dev)
cum_share = np.cumsum(sorted_c) / sorted_c.sum()
x_lorenz = np.linspace(0, 1, len(cum_share))
ax.plot(x_lorenz, cum_share, color="#C44E52", lw=2, label="Lorenz curve")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect equality")
ax.fill_between(x_lorenz, cum_share, x_lorenz, alpha=0.15, color="#C44E52")

# Gini coefficient from Lorenz
gini_global = 1 - 2 * np.trapezoid(cum_share, x_lorenz)
ax.text(0.15, 0.85, f"Global Gini = {gini_global:.3f}",
        fontsize=11, transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="grey"))
ax.set_xlabel("Cumulative Share of Developers (ranked)"); ax.set_ylabel("Cumulative Share of Commits")
ax.set_title("Lorenz Curve: Developer Commits (All Repos)")
ax.legend(fontsize=9)

# Panel C: log-log rank-frequency of commits
ax = axes[2]
sorted_desc = np.sort(commits_per_dev)[::-1]
ranks = np.arange(1, len(sorted_desc) + 1)
ax.scatter(ranks, sorted_desc, s=3, alpha=0.4, c="#4C72B0")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Developer Rank (log)"); ax.set_ylabel("Total Commits (log)")
ax.set_title("Rank-Frequency Distribution")

# Annotate top 5
for i in range(5):
    login = sorted(dev_commits, key=dev_commits.get, reverse=True)[i]
    ax.annotate(login, (ranks[i], sorted_desc[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5))

plt.tight_layout()
plt.savefig(OUT_DIR / "developer_participation.png", dpi=150)
plt.close()
print("  Saved: developer_participation.png")


# ================================================================
# CHART 7 — Heatmap: Metric correlations
# ================================================================
print("Chart 7: Metric correlation heatmap ...")

corr_cols = ["gini_commits", "hhi_commits", "top1_share", "top5_share",
             "bus_factor_80pct", "n_authors", "total_commits",
             "normalized_entropy", "_stars", "age_years"]
corr_labels = ["Gini", "HHI", "Top-1%", "Top-5%", "Bus Factor",
               "Authors", "Commits", "Norm Entropy", "Stars", "Age (yr)"]

corr_matrix = df[corr_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(10, 8.5))
fig.suptitle("Spearman Rank Correlation Matrix", fontsize=14, fontweight="bold")

im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(corr_labels)))
ax.set_yticklabels(corr_labels, fontsize=10)

# Annotate cells
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_matrix.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=color)

fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved: correlation_heatmap.png")


# ================================================================
print("\n✓ All 7 extra charts saved to output/")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name:<45s} {f.stat().st_size/1024:.0f} KB")
