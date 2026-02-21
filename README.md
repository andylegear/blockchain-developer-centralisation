# Blockchain Developer Centralisation — Reproducibility Pack

[![DOI](https://img.shields.io/badge/Paper-BCCA%202026-blue)](https://doi.org/TBD)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Companion repository** for:
>
> A. R. Sai and A. Le Gear, *"Beyond Commit Counts: A Multi-Dimensional Empirical Analysis of Developer Centralization in Blockchain Open-Source Ecosystems,"* IEEE International Conference on Blockchain Computing and Applications (BCCA), Barcelona, 2026.

This repository contains all code, data, and results needed to fully reproduce the analyses reported in the paper. It does **not** contain the LaTeX source of the paper itself.

---

## Repository Structure

```
.
├── analysis/
│   ├── centralization_analysis.py   # Core analysis (sampling, API fetch, metrics)
│   ├── extended_analysis.py         # Language, age, entropy, cross-pollination
│   ├── extra_charts.py              # 7 multi-dimensional visualisations
│   ├── untapped_analysis.py         # LoC concentration, tenure, arrivals, profiling
│   ├── RESULTS.md                   # Full narrative results (21 sections)
│   ├── EXTENDED_ANALYSES.md         # Description of the extended analysis scripts
│   ├── cache/                       # Cached GitHub API responses (352 JSON files)
│   └── output/                      # All generated CSVs and PNGs
│       ├── *.csv                    # 11 data tables
│       └── *.png                    # 18 figures
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A GitHub personal access token (set as `GITHUB_TOKEN` environment variable, or hard-coded fallback in script)

### Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install pandas numpy matplotlib scipy PyGithub tqdm tenacity
```

### Running the Analyses

The scripts should be run in order. Each subsequent script depends on the outputs of the previous ones.

```bash
cd analysis

# 1. Core analysis — sampling, API data collection, base metrics
python centralization_analysis.py

# 2. Extended cross-sectional analyses — language, age, entropy, cross-pollination
python extended_analysis.py

# 3. Additional visualisations combining multiple dimensions
python extra_charts.py

# 4. LoC concentration, contributor tenure, arrivals/departures, role profiling
python untapped_analysis.py
```

> **Note:** Step 1 makes GitHub API calls and may take 30–60 minutes depending on rate limits. The `cache/` directory contains pre-fetched responses so the analysis can be re-run without API access. Steps 2–4 use only cached data and CSV outputs.

---

## Output Inventory

### Data Tables (CSV)

| File | Description | Script |
|------|-------------|--------|
| `sampled_repos.csv` | 357 stratified-sampled repositories with metadata | `centralization_analysis.py` |
| `centralization_summary.csv` | Per-repo Gini, HHI, top-k, bus factor (commits) | `centralization_analysis.py` |
| `centralization_monthly.csv` | Monthly Gini time series per repo | `centralization_analysis.py` |
| `aggregate_statistics.csv` | Population-level summary statistics | `centralization_analysis.py` |
| `language_vs_centralization.csv` | Per-language concentration breakdown | `extended_analysis.py` |
| `age_vs_centralization.csv` | Age-vs-metric correlations and regression | `extended_analysis.py` |
| `shannon_entropy.csv` | Per-repo Shannon entropy and normalised entropy | `extended_analysis.py` |
| `ecosystem_overlap.csv` | Shared-developer counts between ecosystem pairs | `extended_analysis.py` |
| `top_cross_pollinators.csv` | Developers contributing to 3+ repositories | `extended_analysis.py` |
| `untapped_repo_metrics.csv` | Per-repo LoC Gini, tenure, retention metrics | `untapped_analysis.py` |
| `untapped_contributor_details.csv` | Per-contributor tenure, consistency, role profile | `untapped_analysis.py` |

### Figures (PNG)

| File | Description | Paper Fig. |
|------|-------------|------------|
| `centralization_plots.png` | 6-panel core metrics overview | — |
| `loc_concentration.png` | Commit Gini vs LoC Gini comparison | Fig. 1 |
| `arrival_departure.png` | Monthly contributor arrivals & departures | Fig. 2 |
| `ecosystem_network.png` | Cross-pollination network graph | Fig. 3 |
| `language_vs_centralization.png` | Box plots of metrics by language | — |
| `age_vs_centralization.png` | Scatter + OLS regression (age vs Gini) | — |
| `shannon_entropy.png` | Entropy distribution and correlations | — |
| `cross_pollination.png` | Developer-repo bipartite statistics | — |
| `radar_language_profiles.png` | Spider chart of language profiles | — |
| `bubble_age_gini_stars.png` | Bubble chart (age × Gini × stars) | — |
| `violin_language_metrics.png` | Violin plots by language | — |
| `gini_vs_entropy_by_language.png` | Gini vs entropy scatter | — |
| `correlation_heatmap.png` | Spearman correlation matrix | — |
| `developer_participation.png` | Repos-per-developer histogram | — |
| `tenure_retention.png` | Tenure distribution and retention rates | — |
| `consistency_analysis.png` | Activity consistency vs output | — |
| `authorship_vs_maintenance.png` | Author / Balanced / Maintainer profiles | — |
| `untapped_dashboard.png` | 4-panel LoC & sustainability summary | — |

---

## Methodology Summary

1. **Population**: 4,905 blockchain OSS repositories from the [Awesome Crypto Projects](https://www.awesomecrypto.xyz/) dataset
2. **Sampling**: Stratified by star-count deciles; 357 sampled, 345 successfully retrieved (95% CI, 5% MoE)
3. **Data source**: GitHub REST API `stats/contributors` endpoint (weekly commit, addition, deletion counts per contributor)
4. **Metrics**: Gini coefficient, HHI, top-k shares (k=1,5,10), bus factor (80%), Shannon entropy — computed on both commit counts and lines-of-code churn
5. **Extended analyses**: Language/age/popularity moderators, temporal persistence, contributor tenure & retention, arrival/departure curves, authorship profiling, cross-pollination network

See [RESULTS.md](analysis/RESULTS.md) for the full narrative results and [EXTENDED_ANALYSES.md](analysis/EXTENDED_ANALYSES.md) for a description of each analysis script.

---

## Citation

```bibtex
@inproceedings{sai2026beyondcommits,
  author    = {Sai, Ashish Rajendra and {Le Gear}, Andrew},
  title     = {Beyond Commit Counts: A Multi-Dimensional Empirical Analysis
               of Developer Centralization in Blockchain Open-Source Ecosystems},
  booktitle = {2026 IEEE International Conference on Blockchain Computing
               and Applications (BCCA)},
  year      = {2026},
  publisher = {IEEE},
  address   = {Barcelona, Spain}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).
