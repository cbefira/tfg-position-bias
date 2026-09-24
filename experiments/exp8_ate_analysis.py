"""
EXPERIMENT 8, part 2 -- ATE in ranking, evaluated offline.

WHAT THIS DOES
--------------
Part 1 saved, for every PSC invocation, each individual run's output ranking
together with the input slot each item occupied. Because aggregation and
escalation both happen AFTER the model has answered, Agreement-Triggered
Escalation can be reconstructed from that saved data without a single
additional API call. This script does that, and produces both figures for
Experiment 8.

RUN
---
    python experiments/exp8_ate_analysis.py

Reads results/exp8_mitigation_ranking.json. No API calls, no cost.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stats_utils import (kendall_tau, PLOTS_DIR, C_GRAY, C_RED,
                         C_DBLUE, C_TEAL, C_NAVY, C_BAND, C_GRID)

DEFAULT_THRESHOLD = 0.7
THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
RESULTS_FILE = "exp8_mitigation_ranking.json"


def find_results() -> Path:
    here = Path(__file__).resolve().parent
    for base in (here.parent, here, Path.cwd()):
        candidate = base / "results" / RESULTS_FILE
        if candidate.exists():
            return candidate
    raise SystemExit(f"Could not find results/{RESULTS_FILE}. "
                     f"Run exp8_mitigation_ranking.py first.")


def borda_aggregate(runs: list[dict], items: list[int]) -> list[int]:
    """Borda count over a set of runs: an item in position p of a K-item
    ranking scores K-1-p. This is what PSC returns, and what ATE returns
    when it escalates."""
    k = len(items)
    score: dict[int, float] = defaultdict(float)
    for run in runs:
        for pos, item in enumerate(run["ranking"]):
            score[item] += k - 1 - pos
    return sorted(items, key=lambda it: (-score[it], it))


def simulate_ate(fwd_runs: list[dict], rev_runs: list[dict],
                 items: list[int], threshold: float):
    """Returns (ranking_fwd, ranking_rev, calls_spent, agreed)."""
    probe_f = fwd_runs[0]["ranking"]
    probe_r = rev_runs[0]["ranking"]
    if kendall_tau(probe_f, probe_r) >= threshold:
        return probe_f, probe_r, 2, True
    return (borda_aggregate(fwd_runs, items),
            borda_aggregate(rev_runs, items), len(fwd_runs), False)


def paired_p(a: list[float], b: list[float]) -> float | None:
    """Paired t-test against PSC, or None when the two are identical."""
    if all(abs(x - y) < 1e-12 for x, y in zip(a, b)):
        return None
    try:
        import scipy.stats as sps
        return float(sps.ttest_rel(a, b).pvalue)
    except Exception:
        return float("nan")


def plot_strategy_comparison(summary: dict, ate_pc: list[float],
                             ate_calls: float, k: int) -> None:
    """The four strategies on one axis: what each buys and what it costs.

    Reproduces the figure used in the thesis defence; colours, hatching and
    label placement are fixed so the write-up and the talk show the same
    figure rather than two renderings of the same numbers.
    """
    names, means, errs, costs, colors, hatches = [], [], [], [], [], []
    for key, label, color in (("standard", "Standard\n(baseline)", C_GRAY),
                              ("rise", "RISE@1\n(Bito 2025)", C_RED),
                              ("psc", "PSC / Borda\n(Tang 2024)", C_DBLUE)):
        if key not in summary:
            continue
        names.append(label)
        means.append(summary[key]["mean_pc"])
        errs.append(summary[key]["std_pc"])
        n_calls = summary[key]["calls_per_ranking"]
        costs.append(f"{n_calls} call" + ("s" if n_calls != 1 else ""))
        colors.append(color)
        hatches.append("")
    names.append(f"ATE\n(\u03c4 \u2265 {DEFAULT_THRESHOLD})")
    means.append(st.mean(ate_pc))
    errs.append(st.pstdev(ate_pc) if len(ate_pc) > 1 else 0.0)
    costs.append(f"{ate_calls:.1f} calls")
    colors.append(C_TEAL)
    hatches.append("//")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(names, means, yerr=errs, capsize=5, width=0.62,
                  color=colors, error_kw=dict(ecolor=C_GRAY, lw=1.4))
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor("white")
            bar.set_linewidth(0)
    for i, (m, c) in enumerate(zip(means, costs)):
        ax.text(i, m - 0.055, f"{m:.3f}", ha="center", va="top", color="white",
                fontsize=13, fontweight="bold")
        ax.text(i, 0.028, c, ha="center", color="white", fontsize=10,
                fontweight="bold")
    ax.axhline(1.0, ls=":", color=C_NAVY, lw=1.1)
    ax.text(-0.42, 1.012, "fully order-invariant", fontsize=8.5,
            style="italic", color=C_NAVY)
    ax.set_ylabel("Positional consistency  (Kendall's \u03c4)", fontsize=10)
    ax.set_ylim(0, 1.06)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.grid(axis="y", color=C_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = PLOTS_DIR / "exp8_mitigation_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved plot -> {out}")


def plot_threshold_curve(rows: list[dict], psc_pc: float, k: int) -> None:
    """The whole operating curve: consistency and cost against the threshold.

    Every operating point is plotted, including the low-threshold end where
    ATE is significantly worse than PSC -- that point is marked in red rather
    than quietly omitted.
    """
    thr = [r["threshold"] for r in rows]
    pc = [r["mean_pc"] for r in rows]
    calls = [r["avg_calls"] for r in rows]
    band = (DEFAULT_THRESHOLD - 0.035, DEFAULT_THRESHOLD + 0.035)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.8), sharex=True)

    # --- top: positional consistency -------------------------------------
    ax1.axvspan(*band, color=C_BAND, zorder=0)
    ax1.plot(thr, pc, marker="o", color=C_TEAL, lw=2.6, markersize=8,
             zorder=3)
    ax1.axhline(psc_pc, ls="--", color=C_NAVY, lw=1.8, zorder=2)
    ax1.text(thr[0] + 0.02, psc_pc + 0.004, f"PSC = {psc_pc:.3f}",
             fontsize=11, color=C_NAVY, fontweight="bold", va="bottom")
    ax1.set_ylim(min(pc) - 0.038, max(pc) + 0.042)
    ax1.text(DEFAULT_THRESHOLD, max(pc) + 0.012,
             f"\u03c4 \u2265 {DEFAULT_THRESHOLD}", fontsize=11,
             color=C_NAVY, fontweight="bold", ha="center", va="bottom")
    # mark any operating point that is significantly worse than PSC
    for r in rows:
        p = r["p_vs_psc"]
        if p is not None and p < 0.05 and r["mean_pc"] < psc_pc:
            ax1.plot(r["threshold"], r["mean_pc"], marker="o", markersize=9,
                     color=C_RED, zorder=4)
            ax1.annotate(f"p = {p:.3f}\nworse than PSC",
                         (r["threshold"], r["mean_pc"]),
                         textcoords="offset points", xytext=(4, -28),
                         fontsize=8.5, color=C_RED, linespacing=1.3)
    ax1.set_ylabel("Positional consistency", fontsize=10)
    ax1.grid(axis="y", color=C_GRID, linewidth=0.9)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # --- bottom: average cost --------------------------------------------
    ax2.axvspan(*band, color=C_BAND, zorder=0)
    ax2.bar(thr, calls, width=0.062, color=C_TEAL, zorder=3)
    for x, v in zip(thr, calls):
        ax2.text(x, v + 0.14, f"{v:.1f}", ha="center", fontsize=10.5,
                 fontweight="bold", color=C_NAVY, zorder=4)
    ax2.axhline(k, ls="--", color=C_NAVY, lw=1.8, zorder=2)
    ax2.text(thr[0] + 0.02, k + 0.14, f"PSC = {k}.0 calls", fontsize=11,
             color=C_NAVY, fontweight="bold", va="bottom")
    # the saving at the reported operating point
    chosen = min(rows, key=lambda r: abs(r["threshold"] - DEFAULT_THRESHOLD))
    x_arrow = DEFAULT_THRESHOLD - 0.055
    ax2.annotate("", xy=(x_arrow, k), xytext=(x_arrow, chosen["avg_calls"]),
                 arrowprops=dict(arrowstyle="<->", color=C_NAVY, lw=1.4),
                 zorder=4)
    # The saving is quoted from the call counts as plotted (one decimal),
    # so the label agrees with the numbers printed on the bars. The exact
    # figure, which can differ by under a percentage point when a run was
    # dropped as unparseable, is reported in the console table and the text.
    saving_shown = (1 - round(chosen["avg_calls"], 1) / k) * 100
    ax2.text(x_arrow - 0.012, (k + chosen["avg_calls"]) / 2,
             f"{saving_shown:.0f}% fewer\ncalls", ha="right",
             va="center", fontsize=9, color=C_NAVY, linespacing=1.3)
    ax2.set_ylabel("Average calls per ranking", fontsize=10)
    ax2.set_xlabel("Agreement threshold on the 2-call probe  "
                   "(Kendall's \u03c4)", fontsize=10)
    ax2.set_xticks(thr)
    ax2.set_xticklabels([f"{t:.1f}" for t in thr])
    ax2.set_ylim(0, k + 1.1)
    ax2.grid(axis="y", color=C_GRID, linewidth=0.9)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    out = PLOTS_DIR / "exp8_ate_threshold.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved plot -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="the operating point highlighted in the figures")
    args = ap.parse_args()

    path = find_results()
    print(f"Reading {path}")
    print("Offline re-evaluation of the saved runs -- no API calls made.\n")
    data = json.loads(path.read_text())

    entries = (data.get("raw_runs") or {}).get("psc") or []
    if not entries:
        raise SystemExit("This results file has no saved raw runs. Re-run "
                         "exp8_mitigation_ranking.py, then this analysis.")

    pairs = []
    for entry in entries:
        fwd, rev = entry.get("fwd"), entry.get("rev")
        if fwd and rev:
            items = sorted({it for r in fwd for it in r["ranking"]})
            pairs.append((fwd, rev, items))

    k = len(pairs[0][0])
    summary = data.get("summary", {})
    psc_pc_per_trial = [kendall_tau(borda_aggregate(f, i), borda_aggregate(r, i))
                        for f, r, i in pairs]
    psc_pc = st.mean(psc_pc_per_trial)

    probe_taus = [kendall_tau(f[0]["ranking"], r[0]["ranking"]) for f, r, _ in pairs]
    print(f"Trials: {len(pairs)} | K = {k} | PSC positional consistency = {psc_pc:.3f}")
    print(f"Probe agreement (τ between the two probe rankings): "
          f"min={min(probe_taus):.2f}  median={st.median(probe_taus):.2f}  "
          f"max={max(probe_taus):.2f}")

    # ---- the whole threshold curve ----------------------------------------
    print(f"\n===== ATE across the full threshold range =====")
    print(f"{'Threshold':<12}{'Escalated':<12}{'Calls':<9}{'Saving':<10}"
          f"{'PC':<9}{'vs PSC':<10}{'p'}")
    rows = []
    for thr in THRESHOLDS:
        pc, cost, escalated = [], [], 0
        for fwd, rev, items in pairs:
            r1, r2, c, agreed = simulate_ate(fwd, rev, items, thr)
            pc.append(kendall_tau(r1, r2))
            cost.append(c)
            escalated += not agreed
        mean_pc, avg_calls = st.mean(pc), st.mean(cost)
        p = paired_p(pc, psc_pc_per_trial)
        p_txt = "identical" if p is None else f"{p:.4f}"
        rows.append({"threshold": thr, "escalated": escalated,
                     "n": len(pc), "avg_calls": avg_calls,
                     "saving": 1 - avg_calls / k, "mean_pc": mean_pc,
                     "diff_vs_psc": mean_pc - psc_pc, "p_vs_psc": p,
                     "pc_per_trial": pc})
        print(f"{thr:<12.1f}{f'{escalated}/{len(pc)}':<12}{avg_calls:<9.1f}"
              f"{f'{(1 - avg_calls / k) * 100:.1f}%':<10}{mean_pc:<9.3f}"
              f"{mean_pc - psc_pc:<+10.3f}{p_txt}")

    chosen = min(rows, key=lambda r: abs(r["threshold"] - args.threshold))
    print(f"\nOperating point τ ≥ {chosen['threshold']:.1f}: "
          f"PC {chosen['mean_pc']:.3f} vs PSC {psc_pc:.3f}, "
          f"{chosen['avg_calls']:.1f} calls instead of {k}.0 "
          f"({chosen['saving'] * 100:.1f}% fewer).")
    # The average can sit just below the nominal escalation cost when a run
    # was dropped as unparseable, which is why this is reported from the
    # measured call count rather than from the nominal 2-or-K budget.

    # ---- figures -----------------------------------------------------------
    print()
    plot_strategy_comparison(summary, chosen["pc_per_trial"],
                             chosen["avg_calls"], k)
    plot_threshold_curve(rows, psc_pc, k)

    out = path.parent / "exp8_ate_analysis.json"
    out.write_text(json.dumps({
        "source": path.name, "k": k, "n_trials": len(pairs),
        "psc_mean_pc": psc_pc, "probe_taus": probe_taus,
        "default_threshold": chosen["threshold"],
        "threshold_sweep": [{kk: vv for kk, vv in r.items()
                             if kk != "pc_per_trial"} for r in rows],
    }, indent=2))
    print(f"\nSaved analysis -> {out}")

    print("\nHow to read this:")
    print("* Each row is a different operating point. A row whose PC is "
          "statistically indistinguishable from PSC while spending well "
          "below K calls is a genuine efficiency gain.")
    print("* A row where PC drops significantly is a trade-off, not a free "
          "win. The low-threshold end escalates too rarely and lands there, "
          "which is why the whole curve is reported rather than one row.")
    print("* ATE does not claim to beat PSC on consistency. It targets cost: "
          "the same stability, paid for only on the decisions that are "
          "actually contested.")


if __name__ == "__main__":
    main()
