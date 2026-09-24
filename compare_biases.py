"""
compare_biases.py -- Cross-experiment comparison. Run after Experiments 1, 4
and 5.

Reads the JSON files those experiments saved into ./results and produces the
bias-magnitude figure plus the console table, placing all three non-content
effects on a single scale.


RUN
---
    python compare_biases.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        print(f"  [skip] {p.name} not found -- run that experiment first.")
        return None
    return json.loads(p.read_text())


def main() -> None:
    print("Cross-experiment bias comparison\n")
    bars: dict[str, float] = {}
    notes: list[str] = []

    e1 = load("exp1_single_choice")
    if e1:
        counts = e1["slot_counts"]
        total = sum(counts)
        best = max(counts) / total if total else 0.0
        bars["Position\n(best slot)"] = (best - 1 / len(counts)) * 100
        notes.append(f"Position:   chi2={e1['chi2_position']:.2f}, "
                     f"p={e1['p_position']:.4f}")

    e4 = load("exp4_popularity_cue")
    if e4:
        bars["Popularity\ncue"] = e4["uplift_vs_uniform"] * 100
        notes.append(f"Popularity: P(chosen|cue)={e4['p_chosen_given_cue']:.3f}")

    e5 = load("exp5_verbosity")
    if e5:
        bars["Verbosity\n(long form)"] = e5["uplift_vs_uniform"] * 100
        notes.append(f"Verbosity:  P(chosen|long)={e5['p_chosen_given_long']:.3f}")

    e3 = load("exp3_pairwise_swap")
    flip_rate = (1 - e3["consistency_rate"]) * 100 if e3 else None
    if e3:
        notes.append(f"Pairwise:   flip rate={flip_rate:.1f}%, "
                     f"share A={e3['share_A_answers'] * 100:.1f}%")

    if bars:
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        names = list(bars)
        vals = [bars[n] for n in names]
        ax.bar(names, vals, color=["#2a78d6", "#1d9e75", "#d85a30"][:len(names)],
               width=0.55)
        ax.axhline(0, color="black", linewidth=0.8)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.6 if v >= 0 else -1.6), f"{v:+.1f}",
                    ha="center", fontsize=9)
        ax.set_ylabel("Uplift over uniform baseline (pp)")
        ax.set_title("Bias magnitude comparison (uplift over 1/N)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)
        fig.tight_layout()
        out = PLOTS / "compare_bias_magnitudes.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"\nSaved comparison figure -> {out}")

        print("\n===== BIAS MAGNITUDES =====")
        print(f"{'Bias':<26}{'Effect'}")
        for name, v in bars.items():
            print(f"{name.replace(chr(10), ' '):<26}{v:+.1f} pp")

    print("\n===== SUMMARY =====")
    for line in notes:
        print("  " + line)
    if flip_rate is not None:
        print("\nThe pairwise flip rate uses a different unit (% of decisions "
              "reversed by a swap) and is reported separately from the "
              "uplift bars.")


if __name__ == "__main__":
    main()
