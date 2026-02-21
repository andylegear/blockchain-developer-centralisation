# Extended Analyses

This document describes the three additional analysis scripts added on top of Ashish's original `centralization_analysis.py`. All three scripts operate on existing cached API data and CSV outputs — **no new API calls** are required.

---

## Original Script: `centralization_analysis.py`

Ashish's baseline analysis, recreated from the JPS-CP paper methodology:

- Loads the 4,905-repo population from `github_data.csv`
- Performs stratified sampling by star-count deciles → 357 repos (345 successfully retrieved)
- Fetches per-contributor commit data via the GitHub REST API (`stats/contributors` endpoint)
- Computes per-repo metrics: Gini coefficient, HHI, top-k shares (k=1,5,10), bus factor (80% threshold)
- Generates aggregate statistics and stratified breakdowns by star-count decile
- Produces the main `centralization_plots.png` (6-panel summary)

**Outputs:** `centralization_summary.csv`, `sampled_repos.csv`, `centralization_monthly.csv`, `aggregate_statistics.csv`, `centralization_plots.png`

---

## Added Script 1: `extended_analysis.py`

Four cross-sectional and complementary analyses:

### 1. Language vs Centralization
- Groups repos by primary programming language (languages with ≥5 repos)
- Computes per-language median Gini, HHI, bus factor, and author count
- Runs Kruskal-Wallis H-tests for Gini, HHI, and bus factor across languages
- Performs post-hoc pairwise Mann-Whitney U tests with Bonferroni correction

### 2. Repository Age vs Centralization
- Calculates Spearman rank correlations between repo age and all concentration metrics
- Fits an OLS regression (Gini ~ age)
- Generates a scatter plot with regression line and confidence band

### 3. Shannon Entropy
- Computes Shannon entropy and normalized entropy for each repo's commit distribution
- Correlates entropy against Gini, HHI, top-1 share, bus factor, stars, and author count
- Generates entropy distribution and correlation plots

### 4. Cross-Pollination Network
- Scans all 352 cached contributor JSON files to build a developer × repository bipartite graph
- Identifies cross-pollinators (developers contributing to 2+ repos) and the top multi-project contributors
- Computes ecosystem-pair overlap (shared developer counts between project categories)
- Generates the ecosystem network diagram

**Outputs:** `language_vs_centralization.csv`, `age_vs_centralization.csv`, `shannon_entropy.csv`, `ecosystem_overlap.csv`, `top_cross_pollinators.csv`, `language_vs_centralization.png`, `age_vs_centralization.png`, `shannon_entropy.png`, `cross_pollination.png`, `ecosystem_network.png`

---

## Added Script 2: `extra_charts.py`

Seven multi-dimensional visualisations combining data from the base and extended analyses:

1. **Radar / Spider Chart** (`radar_language_profiles.png`) — normalised centralization profiles per language across 5 metrics
2. **Bubble Chart** (`bubble_age_gini_stars.png`) — repo age vs Gini, sized by stars, coloured by language
3. **Violin Plot** (`violin_language_metrics.png`) — distribution of Gini, HHI, and entropy by top languages
4. **Gini vs Entropy by Language** (`gini_vs_entropy_by_language.png`) — scatter with language-coloured points
5. **Correlation Heatmap** (`correlation_heatmap.png`) — Spearman rank correlations between all metric pairs
6. **Developer Participation** (`developer_participation.png`) — histogram of repos-per-developer from cross-pollination data
7. **Ecosystem Network** (`ecosystem_network.png`) — weighted graph of shared developers between blockchain ecosystems

**Outputs:** 7 PNG charts (listed above)

---

## Added Script 3: `untapped_analysis.py`

Five additional dimensions extracted from the weekly-granularity data in the cached `stats/contributors` JSON files:

### 1. Lines-of-Code Concentration
- Re-computes Gini, HHI, and top-1 share using LoC churn (additions + deletions) instead of commit counts
- Computes separate Gini for additions-only and deletions-only
- Runs a Wilcoxon signed-rank test comparing LoC Gini vs commit Gini across all repos

### 2. Contributor Tenure & Retention
- Calculates each contributor's tenure (weeks between first and last active week)
- Computes per-repo median tenure, one-time contributor rate, and 6-month retention rate
- Correlates tenure metrics against LoC concentration

### 3. Arrival / Departure Curves
- Derives monthly first-appearance (arrival) and last-appearance (departure) counts across all repos
- Computes rolling 3-month averages and net contributor flow
- Identifies the peak arrival month and the onset of net-negative talent flow

### 4. Activity Consistency
- Measures each contributor's consistency (proportion of weeks active within their tenure span)
- Bins contributors by consistency level and compares median commits and LoC output per bin
- Tests whether sporadic contributors produce more output than consistent ones

### 5. Authorship vs Maintenance Profiling
- Classifies each contributor as Author (addition ratio >0.7), Balanced (0.3–0.7), or Maintainer (<0.3)
- Compares median tenure, commits, and LoC across the three profiles
- Quantifies the scarcity of dedicated maintenance specialists

**Outputs:** `untapped_repo_metrics.csv`, `untapped_contributor_details.csv`, `loc_concentration.png`, `tenure_retention.png`, `arrival_departure.png`, `consistency_analysis.png`, `authorship_vs_maintenance.png`, `untapped_dashboard.png`
