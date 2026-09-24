"""
EXPERIMENT 2 -- Listwise ranking.

DESIGN
------
Same pool and same scenario as Experiment 1; the task format is what
changes. Each trial: one random reshuffle, one call, one complete ranking.

Two analyses:
  1. MEAN OUTPUT RANK vs INPUT SLOT. If unbiased, every slot's expected mean
     output rank is (K+1)/2. A downward slope means primacy (early slots
     favoured).
  2. RANK STABILITY across reshuffles, as Kendall's tau between the item
     orderings produced on different trials. tau = 1.0 would mean the model
     ranks the same items identically regardless of presentation order, so
     any value below 1.0 is the share the ordering owes to input order
     rather than to content.

RUN
---
    python experiments/exp2_listwise_ranking.py --trials 60

One API call per trial.
"""

from __future__ import annotations

import argparse
import itertools
import random
import statistics

from common import (LETTERS, get_items, SYSTEM_RANKING, build_ranking_prompt,
                    call_llm, parse_ranking, shuffled_with_map, save_results)
from stats_utils import kendall_tau, mean_rank_by_position, plot_mean_rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = get_items()
    k = len(pool)
    trials_log = []

    print(f"Experiment 2 -- listwise ranking, {args.trials} trials, "
          f"{k} candidates per trial\n")

    for t in range(args.trials):
        ordered = shuffled_with_map(pool, rng)
        raw = call_llm(build_ranking_prompt(ordered), SYSTEM_RANKING, max_tokens=40)
        ranking_slots = parse_ranking(raw, k)
        if ranking_slots is None:
            print(f"  trial {t + 1:>3}: invalid ranking ({raw!r}), skipped")
            continue
        order = [i for i, _ in ordered]                     # item id per slot
        ranking_items = [order[s] for s in ranking_slots]   # items, best first
        trials_log.append({
            "trial": t,
            "order": order,
            "ranking_slots": ranking_slots,
            "ranking_items": ranking_items,
            "raw": raw,
        })
        print(f"  trial {t + 1:>3}: ranking "
              f"{','.join(LETTERS[s] for s in ranking_slots)}")

    # --- analysis 1: mean output rank per input slot ------------------------
    mean_ranks = mean_rank_by_position(trials_log, k)
    print(f"\nMean output rank by input slot "
          f"(unbiased expectation = {(k + 1) / 2:.1f}):")
    for slot, mr in enumerate(mean_ranks):
        print(f"  Slot {LETTERS[slot]} (shown {slot + 1}.): mean rank = {mr:.2f}")
    plot_mean_rank(mean_ranks, "Exp. 2 — Mean output rank vs. input position",
                   "exp2_mean_rank_by_slot.png")

    # --- analysis 2: ranking stability across reshuffles --------------------
    taus = [kendall_tau(a["ranking_items"], b["ranking_items"])
            for a, b in itertools.combinations(trials_log, 2)]
    tau_mean = statistics.mean(taus) if taus else None
    if taus:
        print(f"\nRanking stability across reshuffles ({len(taus)} trial pairs):")
        print(f"  mean Kendall's tau = {tau_mean:.3f}  (1.0 = order-invariant)")
        if len(taus) > 1:
            print(f"  std = {statistics.stdev(taus):.3f}")

    save_results("exp2_listwise_ranking", {
        "trials": trials_log,
        "k_shown_per_trial": k,
        "mean_rank_by_slot": mean_ranks,
        "kendall_tau_mean": tau_mean,
        "kendall_tau_all": taus,
        "kendall_tau_n_pairs": len(taus),
    })

    print("\nHow to read this:")
    print("* A monotonic slope in the mean-rank plot is primacy; a dip in the "
          "middle would instead be the 'lost in the middle' pattern.")
    print("* tau below 1.0 is the part of the ordering that depends on input "
          "order. If content alone drove the ranking, tau would be 1.0.")


if __name__ == "__main__":
    main()
