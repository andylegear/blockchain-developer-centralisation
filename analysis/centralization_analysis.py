#!/usr/bin/env python3
"""
Blockchain Developer Centralization Analysis
=============================================
Recreates the methodology from:
  "Quantifying Developer Centralization in Blockchain Open-Source Ecosystems"
  (JPS-CP paper by Ashish Rajendra Sai)

Computes Gini coefficient, HHI, top-k contribution shares, and bus factor
for contributor commit distributions across blockchain GitHub repositories.

Methodology:
  1. Load 4905-repo population from the source CSV
  2. Stratified sampling by GitHub-star deciles → 357 repos (95% CI, 5% MoE)
  3. Fetch per-contributor commit data via GitHub REST API
  4. Compute centralization metrics per repo
  5. Aggregate and compare to paper-reported values
"""

import os
import sys
import time
import json
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: PyGithub
try:
    from github import Github, GithubException, RateLimitExceededException
    try:
        from github import Auth as GithubAuth
    except ImportError:
        GithubAuth = None
    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False
    GithubAuth = None
    print("WARNING: PyGithub not installed. Install with: pip install PyGithub")

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
SOURCE_CSV  = SCRIPT_DIR.parent / "sources" / "github_data.csv"
OUT_DIR     = SCRIPT_DIR / "output"
CACHE_DIR   = SCRIPT_DIR / "cache"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Sampling parameters (matching paper §2.1)
Z_SCORE           = 1.96    # 95% confidence
MARGIN_OF_ERROR   = 0.05
POP_PROPORTION    = 0.5
N_STRATA          = 10      # star-count decile bins

# Analysis parameters
BUS_FACTOR_THRESHOLD    = 0.80
MAX_FETCH_TIME_SECONDS  = 3600  # 60-minute cap on API fetching
STATS_POLL_ATTEMPTS     = 6     # retries when GitHub is computing stats

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ============================================================
# Metric Functions
# ============================================================

def gini(array) -> float:
    """Gini coefficient for an array of non-negative values (Dorfman 1979)."""
    arr = np.array(array, dtype=float)
    if arr.size == 0:
        return np.nan
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def hhi(counts) -> float:
    """Herfindahl-Hirschman Index from raw counts (Rhoades 1993)."""
    s = np.array(counts, dtype=float)
    if s.size == 0 or s.sum() <= 0:
        return np.nan
    s = s / s.sum()
    return float(np.sum(s ** 2))


def topk_share(counts, k: int) -> float:
    """Proportion of total contributed by the top-k contributors."""
    arr = np.sort(np.array(counts, dtype=float))[::-1]
    total = arr.sum()
    if total == 0:
        return 0.0
    return float(arr[:k].sum() / total)


def bus_factor(counts, threshold: float = 0.80) -> int:
    """Minimum contributors covering >= threshold of total commits."""
    arr = np.sort(np.array(counts, dtype=float))[::-1]
    total = arr.sum()
    if total == 0:
        return np.nan
    cum = np.cumsum(arr) / total
    return int(np.searchsorted(cum, threshold) + 1)


# ============================================================
# Sampling (§2.1)
# ============================================================

def compute_sample_size(N: int) -> int:
    """Z-score sample size with finite-population correction."""
    z, e, p = Z_SCORE, MARGIN_OF_ERROR, POP_PROPORTION
    n_inf = (z**2 * p * (1 - p)) / e**2
    n_adj = n_inf / (1 + (n_inf - 1) / N)
    return int(np.ceil(n_adj))


def stratified_sample(df: pd.DataFrame, n_sample: int, seed: int = 42) -> pd.DataFrame:
    """Stratified sampling by star-count decile bins."""
    df = df.copy()
    df["star_bin"] = pd.qcut(
        df["_stars"], q=N_STRATA, labels=False, duplicates="drop"
    )
    n_bins = df["star_bin"].nunique()
    bin_counts = df["star_bin"].value_counts().sort_index()
    total = len(df)

    # Proportional allocation per stratum
    allocations = {}
    for b in bin_counts.index:
        alloc = max(1, int(np.round(bin_counts[b] / total * n_sample)))
        allocations[b] = alloc

    # Adjust to hit exact target
    diff = n_sample - sum(allocations.values())
    sorted_bins = bin_counts.sort_values(ascending=False).index.tolist()
    i = 0
    while diff != 0:
        b = sorted_bins[i % len(sorted_bins)]
        if diff > 0:
            allocations[b] += 1
            diff -= 1
        elif allocations[b] > 1:
            allocations[b] -= 1
            diff += 1
        i += 1

    rng = np.random.default_rng(seed)
    sampled = []
    for b, n in allocations.items():
        stratum = df[df["star_bin"] == b]
        n_take = min(n, len(stratum))
        chosen = stratum.sample(n=n_take, random_state=int(rng.integers(1e6)))
        sampled.append(chosen)

    return pd.concat(sampled, ignore_index=True)


# ============================================================
# GitHub API Helpers
# ============================================================

def _cache_path(owner_repo: str, suffix: str) -> Path:
    return CACHE_DIR / f"{owner_repo.replace('/', '__')}.{suffix}.json"


def _to_epoch(x) -> int:
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, dt.datetime):
        if x.tzinfo is None:
            x = x.replace(tzinfo=dt.timezone.utc)
        return int(x.timestamp())
    try:
        return int(x)
    except Exception:
        return 0


def fetch_contributor_stats(gh_client, owner_repo: str) -> Optional[List[Dict]]:
    """Fetch stats/contributors, with caching and polling."""
    cp = _cache_path(owner_repo, "stats_contributors")
    if cp.exists():
        with open(cp) as f:
            return json.load(f)

    if gh_client is None:
        return None

    for attempt in range(STATS_POLL_ATTEMPTS):
        try:
            repo_obj = gh_client.get_repo(owner_repo)
            stats = repo_obj.get_stats_contributors()

            if stats is None:
                # GitHub is lazily computing — wait and retry
                time.sleep(4)
                continue

            result = []
            for s in stats:
                author = getattr(s, "author", None)
                login = getattr(author, "login", None) if author else None
                total = int(getattr(s, "total", 0) or 0)
                weeks = []
                for w in (getattr(s, "weeks", []) or []):
                    weeks.append({
                        "w": _to_epoch(getattr(w, "w", 0)),
                        "a": int(getattr(w, "a", 0) or 0),
                        "d": int(getattr(w, "d", 0) or 0),
                        "c": int(getattr(w, "c", 0) or 0),
                    })
                result.append({
                    "author": {"login": login},
                    "total": total,
                    "weeks": weeks,
                })

            with open(cp, "w") as f:
                json.dump(result, f)
            return result

        except RateLimitExceededException:
            rl = gh_client.get_rate_limit()
            core_rl = getattr(rl, 'core', rl.rate) if hasattr(rl, 'core') else rl.rate
            reset = core_rl.reset.replace(tzinfo=None)
            wait = max(5, (reset - dt.datetime.utcnow()).total_seconds())
            if wait > 120:
                print(f"\n    Rate-limit reset in {wait:.0f}s — stopping fetch.")
                return None
            print(f"\n    Rate limited, waiting {wait:.0f}s...", end=" ", flush=True)
            time.sleep(wait)

        except GithubException as e:
            if e.status == 404:
                print("404", end=" ", flush=True)
            else:
                print(f"ERR({e.status})", end=" ", flush=True)
            return None

        except Exception as e:
            print(f"ERR({e})", end=" ", flush=True)
            return None

    return None


def fetch_commits_fallback(gh_client, owner_repo: str) -> Optional[List[Dict]]:
    """Fallback: recent 12-month commits when stats/contributors fails."""
    cp = _cache_path(owner_repo, "commits_12m")
    if cp.exists():
        with open(cp) as f:
            return json.load(f)

    if gh_client is None:
        return None

    since = dt.datetime.utcnow() - dt.timedelta(days=365)
    commits = []
    try:
        repo_obj = gh_client.get_repo(owner_repo)
        for c in repo_obj.get_commits(since=since):
            commits.append({
                "sha": c.sha,
                "date": c.commit.author.date.isoformat()
                        if c.commit and c.commit.author else None,
                "author_login": c.author.login if c.author else None,
                "author_name": c.commit.author.name
                               if c.commit and c.commit.author else None,
            })
            if len(commits) >= 2000:  # safety cap
                break
        with open(cp, "w") as f:
            json.dump(commits, f)
    except Exception as e:
        print(f"commit-fallback-ERR({e})", end=" ", flush=True)
    return commits if commits else None


# ============================================================
# Per-repo metric computation
# ============================================================

def compute_repo_metrics(stats: List[Dict]) -> tuple:
    """Return (metrics_dict, list_of_monthly_dicts) from contributor stats."""
    author_totals: Dict[str, int] = {}
    monthly_counts: Dict[tuple, int] = {}

    for a in stats:
        login = (a.get("author") or {}).get("login") or "unknown"
        total = a.get("total", 0)
        author_totals[login] = author_totals.get(login, 0) + total

        for w in a.get("weeks", []):
            ts = w.get("w", 0)
            if ts > 0:
                d = dt.datetime.utcfromtimestamp(ts)
                ym = f"{d.year:04d}-{d.month:02d}"
                monthly_counts[(ym, login)] = (
                    monthly_counts.get((ym, login), 0) + int(w.get("c", 0))
                )

    counts = list(author_totals.values())
    total_commits = int(sum(counts))
    n_authors = int(sum(1 for c in counts if c > 0))

    metrics = {
        "total_commits":    total_commits,
        "n_authors":        n_authors,
        "gini_commits":     gini(counts),
        "hhi_commits":      hhi(counts),
        "top1_share":       topk_share(counts, 1),
        "top5_share":       topk_share(counts, 5),
        "top10_share":      topk_share(counts, 10),
        "bus_factor_80pct": bus_factor(counts, BUS_FACTOR_THRESHOLD),
    }

    # Monthly breakdown
    by_month: Dict[str, list] = {}
    for (ym, _), cnt in monthly_counts.items():
        by_month.setdefault(ym, []).append(cnt)

    monthly = [
        {
            "month": ym,
            "gini_commits_month":   gini(arr),
            "total_commits_month":  int(sum(arr)),
            "n_authors_month":      int(sum(1 for x in arr if x > 0)),
        }
        for ym, arr in by_month.items()
    ]

    return metrics, monthly


def compute_from_commits(commits: List[Dict]) -> tuple:
    """Compute metrics from raw commit list (fallback path)."""
    author_totals: Dict[str, int] = {}
    monthly_counts: Dict[tuple, int] = {}

    for c in commits:
        login = c.get("author_login") or c.get("author_name") or "unknown"
        author_totals[login] = author_totals.get(login, 0) + 1
        dstr = c.get("date")
        if dstr:
            d = dt.datetime.fromisoformat(dstr.replace("Z", "+00:00"))
            ym = f"{d.year:04d}-{d.month:02d}"
            monthly_counts[(ym, login)] = monthly_counts.get((ym, login), 0) + 1

    counts = list(author_totals.values())
    total_commits = int(sum(counts))
    n_authors = int(sum(1 for c in counts if c > 0))

    metrics = {
        "total_commits":    total_commits,
        "n_authors":        n_authors,
        "gini_commits":     gini(counts),
        "hhi_commits":      hhi(counts),
        "top1_share":       topk_share(counts, 1),
        "top5_share":       topk_share(counts, 5),
        "top10_share":      topk_share(counts, 10),
        "bus_factor_80pct": bus_factor(counts, BUS_FACTOR_THRESHOLD),
    }

    by_month: Dict[str, list] = {}
    for (ym, _), cnt in monthly_counts.items():
        by_month.setdefault(ym, []).append(cnt)

    monthly = [
        {
            "month": ym,
            "gini_commits_month":   gini(arr),
            "total_commits_month":  int(sum(arr)),
            "n_authors_month":      int(sum(1 for x in arr if x > 0)),
        }
        for ym, arr in by_month.items()
    ]

    return metrics, monthly


# ============================================================
# Plotting
# ============================================================

def generate_plots(summary_df: pd.DataFrame, monthly_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Developer Centralization Metrics (n={len(summary_df)} repos)",
        fontsize=14, fontweight="bold",
    )

    # 1 — Gini histogram
    ax = axes[0, 0]
    ax.hist(summary_df["gini_commits"].dropna(), bins=20, edgecolor="black", alpha=0.7, color="#4C72B0")
    med = summary_df["gini_commits"].median()
    ax.axvline(med, color="red", ls="--", label=f"Median = {med:.2f}")
    ax.set_title("Gini Coefficient"); ax.set_xlabel("Gini"); ax.set_ylabel("Repos"); ax.legend()

    # 2 — HHI histogram
    ax = axes[0, 1]
    ax.hist(summary_df["hhi_commits"].dropna(), bins=20, edgecolor="black", alpha=0.7, color="#55A868")
    med = summary_df["hhi_commits"].median()
    ax.axvline(med, color="red", ls="--", label=f"Median = {med:.2f}")
    ax.set_title("HHI"); ax.set_xlabel("HHI"); ax.set_ylabel("Repos"); ax.legend()

    # 3 — Bus Factor histogram
    ax = axes[0, 2]
    bf = summary_df["bus_factor_80pct"].dropna()
    bins = range(1, int(bf.max()) + 2) if len(bf) > 0 else 10
    ax.hist(bf, bins=bins, edgecolor="black", alpha=0.7, color="#C44E52")
    med = bf.median()
    ax.axvline(med, color="blue", ls="--", label=f"Median = {med:.0f}")
    ax.set_title("Bus Factor (80%)"); ax.set_xlabel("Bus Factor"); ax.set_ylabel("Repos"); ax.legend()

    # 4 — Gini vs #Authors scatter
    ax = axes[1, 0]
    ax.scatter(summary_df["n_authors"], summary_df["gini_commits"], alpha=0.5, s=20, color="#8172B2")
    ax.set_title("Gini vs Number of Authors")
    ax.set_xlabel("Number of Authors"); ax.set_ylabel("Gini")

    # 5 — Top-k boxplot
    ax = axes[1, 1]
    topk_data = [
        summary_df["top1_share"].dropna(),
        summary_df["top5_share"].dropna(),
        summary_df["top10_share"].dropna(),
    ]
    bp = ax.boxplot(topk_data, labels=["Top-1", "Top-5", "Top-10"], patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_title("Top-k Contribution Shares"); ax.set_ylabel("Share of Commits")

    # 6 — Monthly Gini trend
    ax = axes[1, 2]
    if not monthly_df.empty:
        agg = monthly_df.groupby("month")["gini_commits_month"].median().sort_index()
        if len(agg) > 36:
            agg = agg.tail(36)
        ax.plot(range(len(agg)), agg.values, marker="o", markersize=3, color="#CCB974")
        ax.set_title("Monthly Median Gini")
        ax.set_xlabel("Month"); ax.set_ylabel("Median Gini")
        ticks = list(range(0, len(agg), max(1, len(agg) // 6)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([agg.index[i] for i in ticks], rotation=45, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No monthly data", ha="center", va="center", fontsize=12)
        ax.set_title("Monthly Median Gini")

    plt.tight_layout()
    path = out_dir / "centralization_plots.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 65)
    print("  BLOCKCHAIN DEVELOPER CENTRALIZATION ANALYSIS")
    print("  Reproducing methodology from JPS-CP paper")
    print("=" * 65)

    # ── Step 1: Load population ──────────────────────────────
    print(f"\n[1/5] Loading population from {SOURCE_CSV.name}")
    df = pd.read_csv(SOURCE_CSV)
    N = len(df)
    print(f"      Population size: {N} repositories")

    # Derive owner/repo from GitHub URL
    def parse_owner_repo(row):
        url = str(row.get("githuburl", "")).rstrip("/")
        parts = url.split("/")
        if len(parts) >= 2 and parts[-2] and parts[-1]:
            return f"{parts[-2]}/{parts[-1]}"
        return f"{row.get('_organization', '')}/{row.get('_reponame', '')}"

    df["owner_repo"] = df.apply(parse_owner_repo, axis=1)

    # ── Step 2: Stratified sampling ──────────────────────────
    print("\n[2/5] Computing sample size and stratification")
    n_sample = compute_sample_size(N)
    print(f"      Formula: n = Z²·p·(1-p)/E²  with FPC")
    print(f"      Z={Z_SCORE}, E={MARGIN_OF_ERROR}, p={POP_PROPORTION}")
    print(f"      n_infinite = {(Z_SCORE**2 * POP_PROPORTION * (1-POP_PROPORTION)) / MARGIN_OF_ERROR**2:.1f}")
    print(f"      n_adjusted (N={N}) = {n_sample}")

    sample_df = stratified_sample(df, n_sample)
    print(f"      Sampled: {len(sample_df)} repos across {sample_df['star_bin'].nunique()} strata")

    sample_path = OUT_DIR / "sampled_repos.csv"
    sample_df.to_csv(sample_path, index=False)

    print("\n      Stratum breakdown:")
    for b in sorted(sample_df["star_bin"].unique()):
        sub = sample_df[sample_df["star_bin"] == b]
        print(f"        Bin {b:2d}: {len(sub):4d} repos  "
              f"(stars {sub['_stars'].min():>7,.0f} – {sub['_stars'].max():>7,.0f})")

    # ── Step 3: Fetch contributor data ───────────────────────
    print("\n[3/5] Fetching contributor data from GitHub API")

    gh_client = None
    if HAS_GITHUB:
        if GITHUB_TOKEN:
            try:
                if GithubAuth:
                    gh_client = Github(auth=GithubAuth.Token(GITHUB_TOKEN), per_page=100)
                else:
                    gh_client = Github(GITHUB_TOKEN, per_page=100)
                rl = gh_client.get_rate_limit()
                core_rl = getattr(rl, 'core', rl.rate) if hasattr(rl, 'core') else rl.rate
                print(f"      Authenticated — remaining: {core_rl.remaining}/{core_rl.limit}")
            except Exception as e:
                print(f"      Auth failed ({e}), falling back to unauthenticated")
                gh_client = None

        if gh_client is None:
            try:
                gh_client = Github(per_page=100)
                rl = gh_client.get_rate_limit()
                core_rl = getattr(rl, 'core', rl.rate) if hasattr(rl, 'core') else rl.rate
                print(f"      Unauthenticated — remaining: {core_rl.remaining}/{core_rl.limit}")
            except Exception as e:
                print(f"      Cannot reach GitHub API: {e}")

    summary_rows = []
    monthly_rows = []
    fetch_start = time.time()
    ok, skip = 0, 0
    repos = sample_df["owner_repo"].tolist()

    for i, owner_repo in enumerate(repos):
        elapsed = time.time() - fetch_start
        if elapsed > MAX_FETCH_TIME_SECONDS:
            print(f"\n      ⏱ Time cap ({MAX_FETCH_TIME_SECONDS}s) reached after {i} repos")
            break

        print(f"  [{i+1:3d}/{len(repos)}] {owner_repo:<45s}", end=" ", flush=True)

        # Primary: stats/contributors
        stats = fetch_contributor_stats(gh_client, owner_repo)
        if stats and len(stats) > 0:
            metrics, monthly = compute_repo_metrics(stats)
        else:
            # Fallback: recent commits
            commits = fetch_commits_fallback(gh_client, owner_repo)
            if commits and len(commits) > 0:
                metrics, monthly = compute_from_commits(commits)
            else:
                skip += 1
                print("SKIP")
                continue

        metrics["owner_repo"] = owner_repo
        summary_rows.append(metrics)
        for m in monthly:
            m["owner_repo"] = owner_repo
        monthly_rows.extend(monthly)
        ok += 1
        print(f"✓  authors={metrics['n_authors']:4d}  "
              f"Gini={metrics['gini_commits']:.3f}  "
              f"HHI={metrics['hhi_commits']:.3f}  "
              f"BF={metrics['bus_factor_80pct']}")

    print(f"\n      Fetched: {ok} repos, skipped: {skip}")

    if not summary_rows:
        print("\n  ✗ No contributor data could be retrieved.")
        print("    Set GITHUB_TOKEN env var with a valid GitHub PAT and re-run.")
        print("    Example:  $env:GITHUB_TOKEN='ghp_...'; python centralization_analysis.py")
        sys.exit(1)

    # ── Step 4: Aggregate results ────────────────────────────
    print("\n[4/5] Computing aggregate statistics")

    summary_df = pd.DataFrame(summary_rows)
    monthly_df = pd.DataFrame(monthly_rows)

    summary_df.to_csv(OUT_DIR / "centralization_summary.csv", index=False)
    monthly_df.to_csv(OUT_DIR / "centralization_monthly.csv",  index=False)

    metric_cols = [
        "gini_commits", "hhi_commits", "top1_share", "top5_share",
        "top10_share", "bus_factor_80pct", "n_authors", "total_commits",
    ]

    print(f"\n      {'Metric':<22s} {'Median':>9s} {'Mean':>9s} {'Std':>9s} {'Min':>9s} {'Max':>9s}")
    print(f"      {'-'*58}")
    agg_rows = []
    for col in metric_cols:
        row = {
            "metric": col,
            "median": summary_df[col].median(),
            "mean":   summary_df[col].mean(),
            "std":    summary_df[col].std(),
            "min":    summary_df[col].min(),
            "max":    summary_df[col].max(),
        }
        agg_rows.append(row)
        print(f"      {col:<22s} {row['median']:>9.4f} {row['mean']:>9.4f} "
              f"{row['std']:>9.4f} {row['min']:>9.4f} {row['max']:>9.4f}")

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "aggregate_statistics.csv", index=False)

    # Comparison with paper
    print(f"\n      Paper comparison (n_repos_analysed = {len(summary_df)}):")
    print(f"      {'Metric':<25s} {'Paper':>9s} {'This Run':>9s}")
    print(f"      {'-'*43}")
    comparisons = [
        ("Median Gini",       0.82, summary_df["gini_commits"].median()),
        ("Median HHI",        0.34, summary_df["hhi_commits"].median()),
        ("Mean Top-1 Share",  0.35, summary_df["top1_share"].mean()),
        ("Mean Top-5 Share",  0.78, summary_df["top5_share"].mean()),
        ("Median Bus Factor", 4.00, summary_df["bus_factor_80pct"].median()),
    ]
    for name, paper, ours in comparisons:
        print(f"      {name:<25s} {paper:>9.2f} {ours:>9.2f}")

    # ── Step 5: Plots ────────────────────────────────────────
    print("\n[5/5] Generating visualisations")
    plot_path = generate_plots(summary_df, monthly_df, OUT_DIR)
    print(f"      Saved: {plot_path}")

    # ── Done ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  ANALYSIS COMPLETE  ({elapsed:.1f}s)")
    print(f"  Repos analysed: {len(summary_df)} / {n_sample} sampled")
    print(f"  Output directory: {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    • {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
