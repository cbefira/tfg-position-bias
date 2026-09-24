"""
EXPERIMENT 6 -- Does model capability help?

DESIGN
------
The exact same single-choice protocol as Experiment 1 (same item pool,
same scenario, same prompt template, same metrics) run once per model in
common.MODEL_CATALOG, which holds three capability tiers of the same vendor
and generation (Claude Haiku 4.5, Sonnet 5, Opus 5). Only the model changes,
which is what makes the comparison fair. Results are reported smallest to
largest.

RUN
---
    python experiments/exp6_model_comparison.py --trials 30

Needs ANTHROPIC_API_KEY. Calls = --trials x len(MODEL_CATALOG).
"""

from __future__ import annotations

import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (LETTERS, MODEL_CATALOG, get_items, SYSTEM_SINGLE,
                    build_single_pick_prompt, call_llm, parse_single_letter,
                    shuffled_with_map, save_results)
from stats_utils import (chi_square_uniform, shannon_entropy, selection_rates,
                         PLOTS_DIR)


def run_one_model(entry: dict, pool: list[str], trials: int,
                  rng: random.Random) -> dict:
    k = len(pool)
    slot_counts = [0] * k
    trials_log = []
    print(f"\n--- {entry['label']} ({entry['provider']}:{entry['model']}) ---")

    for t in range(trials):
        ordered = shuffled_with_map(pool, rng)
        try:
            raw = call_llm(build_single_pick_prompt(ordered), SYSTEM_SINGLE,
                           max_tokens=5, provider=entry["provider"],
                           model=entry["model"])
        except SystemExit as e:
            print(f"  trial {t + 1:>3}: call failed ({e}), skipped")
            continue
        slot = parse_single_letter(raw, k)
        if slot is None:
            print(f"  trial {t + 1:>3}: unparseable ({raw!r}), skipped")
            continue
        slot_counts[slot] += 1
        trials_log.append({"trial": t, "chosen_slot": slot,
                           "order": [i for i, _ in ordered]})
        if (t + 1) % 10 == 0 or t == trials - 1:
            print(f"  trial {t + 1:>3}/{trials}: slot {LETTERS[slot]}")

    chi2, p = chi_square_uniform(slot_counts)
    ent = shannon_entropy(slot_counts)
    rates = selection_rates(slot_counts)
    uplift = (max(rates) if rates else 0.0) - 1 / k
    print(f"  -> chi2={chi2:.2f}  p={p:.4f}  entropy={ent:.3f}  "
          f"best-slot uplift={uplift * 100:+.1f} pp")
    return {
        "label": entry["label"], "provider": entry["provider"],
        "model": entry["model"], "tier": entry["tier"],
        "slot_counts": slot_counts, "chi2": chi2, "p_value": p,
        "entropy": ent, "max_slot_uplift": uplift, "trials": trials_log,
    }


def plot_comparison(results: list[dict]) -> None:
    labels = [r["label"] for r in results]
    panels = [
        ([r["chi2"] for r in results], "Chi-square (higher = more biased)", "#2a78d6"),
        ([r["entropy"] for r in results], "Choice entropy, bits (lower = more biased)", "#1d9e75"),
        ([r["max_slot_uplift"] * 100 for r in results], "Best-slot uplift, pp (higher = more biased)", "#d85a30"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (vals, title, color) in zip(axes, panels):
        ax.bar(labels, vals, color=color, width=0.55)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.suptitle("Exp. 6 — Position bias across Claude capability tiers "
                 "(smallest → largest)", fontsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "exp6_model_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nSaved plot -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30,
                    help="trials PER MODEL (total calls = trials x catalog size)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = get_items()
    catalog = sorted(MODEL_CATALOG, key=lambda e: e["tier"])

    print(f"Experiment 6 -- position bias across {len(catalog)} capability "
          f"tiers, {args.trials} trials each "
          f"({args.trials * len(catalog)} calls total)")
    for e in catalog:
        print(f"  tier {e['tier']}  {e['label']}  ({e['provider']}:{e['model']})")

    results = [run_one_model(e, pool, args.trials, rng) for e in catalog]
    plot_comparison(results)

    save_results("exp6_model_comparison", {
        "k_shown_per_trial": len(pool),
        "trials_per_model": args.trials,
        "models": results,
    }, models_under_test=[{"label": e["label"], "provider": e["provider"],
                           "model": e["model"]} for e in catalog])

    print("\n===== SUMMARY (smallest -> largest) =====")
    print(f"{'Model':<20}{'Chi2':<9}{'p-value':<11}{'Entropy':<10}{'Uplift(pp)'}")
    for r in results:
        print(f"{r['label']:<20}{r['chi2']:<9.2f}{r['p_value']:<11.4f}"
              f"{r['entropy']:<10.3f}{r['max_slot_uplift'] * 100:<+10.1f}")

    print("\nHow to read this:")
    print("* A falling chi-square and a rising entropy across the tiers is "
          "attenuation, not elimination.")
    print("* A p-value above 0.05 at the top tier indicates a smaller effect "
          "at this sample size, not a proven absence of bias.")


if __name__ == "__main__":
    main()
