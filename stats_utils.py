"""
stats_utils.py -- Metrics and plotting helpers shared by all experiments.


"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def selection_rates(counts: list[int]) -> list[float]:
    total = sum(counts)
    return [c / total if total else 0.0 for c in counts]


def chi_square_uniform(counts: list[int]) -> tuple[float, float]:
    """Chi-square goodness-of-fit against a uniform distribution -> (chi2, p).
    Under the null hypothesis of no position bias, every slot is picked with
    probability 1/K."""
    total, n = sum(counts), len(counts)
    if total == 0:
        return 0.0, 1.0
    chi2, p = sps.chisquare(counts, f_exp=[total / n] * n)
    return float(chi2), float(p)


def shannon_entropy(counts: list[int]) -> float:
    """Entropy of a choice distribution, in bits. Lower = more concentrated,
    i.e. the pipeline converges on one answer instead of varying with order."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def max_entropy(n: int) -> float:
    return math.log2(n)


def kendall_tau(rank_a: list[int], rank_b: list[int]) -> float:
    """Kendall's tau between two rankings of the same items, each given as a
    list of item ids in ranked order. 1.0 = identical ordering."""
    pos_a = {item: i for i, item in enumerate(rank_a)}
    pos_b = {item: i for i, item in enumerate(rank_b)}
    items = list(pos_a)
    tau, _ = sps.kendalltau([pos_a[i] for i in items], [pos_b[i] for i in items])
    return float(tau)


def mean_rank_by_position(trials: list[dict], n_items: int) -> list[float]:
    """Average output rank an item receives as a function of the input slot
    it was shown in. Unbiased expectation is (K+1)/2 for every slot; a
    downward slope is primacy, a dip in the middle is 'lost in the middle'.

    Trial entries must contain 'ranking_slots' (slots in ranked order).
    """
    sums = [0.0] * n_items
    counts = [0] * n_items
    for t in trials:
        for out_rank, slot in enumerate(t["ranking_slots"]):
            sums[slot] += out_rank + 1          # 1-based rank
            counts[slot] += 1
    return [s / c if c else float("nan") for s, c in zip(sums, counts)]


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

_BLUE = "#2a78d6"
_TEAL = "#1d9e75"
_CORAL = "#d85a30"
_GRAY = "#888780"

# Palette for the mitigation figures (Experiments 7 and 8). 
C_GRAY = "#5a6570"    # unmitigated baseline / standard
C_RED = "#b85042"     # debias prompt / RISE
C_DBLUE = "#065a82"   # PSC
C_TEAL = "#1c7293"    # ATE (proposed)
C_NAVY = "#21295c"    # annotations, reference lines, value labels
C_BAND = "#e2e9ee"    # shaded highlight band
C_GRID = "#e0e7ec"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)


def _save(fig, filename: str) -> Path:
    out = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved plot -> {out}")
    return out


def plot_selection_rates(counts: list[int], labels: list[str], title: str,
                         filename: str, baseline: float | None = None,
                         xlabel: str = "Position slot in prompt") -> Path:
    rates = [r * 100 for r in selection_rates(counts)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(labels, rates, color=_BLUE, width=0.62)
    if baseline is not None:
        ax.axhline(baseline * 100, color=_CORAL, linewidth=1.2, linestyle="--",
                   label=f"Uniform baseline ({baseline * 100:.1f}%)")
        ax.legend(fontsize=9, frameon=False)
    _style(ax, title, xlabel, "Selection rate (%)")
    return _save(fig, filename)


def plot_grouped_rates(groups: dict[str, list[int]], labels: list[str],
                       title: str, filename: str) -> Path:
    """Two or more series of counts side by side."""
    colors = [_BLUE, _TEAL, _CORAL, _GRAY]
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for gi, (gname, counts) in enumerate(groups.items()):
        rates = [r * 100 for r in selection_rates(counts)]
        xs = [i + gi * width - 0.4 + width / 2 for i in range(len(labels))]
        ax.bar(xs, rates, width=width * 0.92, color=colors[gi % len(colors)],
               label=gname)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9, frameon=False)
    _style(ax, title, "", "Rate (%)")
    return _save(fig, filename)


def plot_probability_bars(probs: list[float], labels: list[str], title: str,
                          filename: str, baseline: float | None = None) -> Path:
    """Bars of raw probabilities (not normalised counts)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    colors = [_BLUE, _GRAY][:len(probs)] if len(probs) <= 2 else _BLUE
    ax.bar(labels, [p * 100 for p in probs], color=colors, width=0.5)
    if baseline is not None:
        ax.axhline(baseline * 100, color=_CORAL, linewidth=1.2, linestyle="--",
                   label=f"Uniform baseline ({baseline * 100:.1f}%)")
        ax.legend(fontsize=9, frameon=False)
    _style(ax, title, "", "Selection probability (%)")
    return _save(fig, filename)


def plot_mean_rank(mean_ranks: list[float], title: str, filename: str) -> Path:
    n = len(mean_ranks)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = list(range(1, n + 1))
    ax.plot(xs, mean_ranks, marker="o", color=_BLUE, linewidth=2)
    ax.axhline((n + 1) / 2, color=_CORAL, linewidth=1.2, linestyle="--",
               label=f"Unbiased expectation ({(n + 1) / 2:.1f})")
    ax.set_xticks(xs)
    ax.invert_yaxis()               # rank 1 (best) at the top
    ax.legend(fontsize=9, frameon=False)
    _style(ax, title, "Input slot (1 = shown first)", "Mean output rank (1 = best)")
    return _save(fig, filename)


def print_summary(name: str, counts: list[int], labels: list[str]) -> None:
    chi2, p = chi_square_uniform(counts)
    ent = shannon_entropy(counts)
    n, total = len(counts), sum(counts)
    print(f"\n===== {name} =====")
    print(f"{'':<10}{'Chosen':<10}{'Rate':<10}")
    for lab, c in zip(labels, counts):
        print(f"{lab:<10}{c:<10}{(c / total * 100 if total else 0):>5.1f}%")
    print(f"\nChi-square = {chi2:.2f}   p-value = {p:.4f}   "
          f"(df = {n - 1}; p < 0.05 => significant deviation from uniform)")
    print(f"Choice entropy = {ent:.3f} bits (max = {max_entropy(n):.3f}; "
          f"lower = more concentrated)")
