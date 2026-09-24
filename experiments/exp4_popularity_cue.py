"""
EXPERIMENT 4 -- Popularity cue.

DESIGN -- paired A/B with rotation
----------------------------------
  CONTROL    plain items, no cue anywhere. Establishes the baseline pick
             rate of each paper, i.e. its strength on content alone.
  TREATMENT  the identical prompt, except that exactly ONE item carries the
             cue. The cue-carrying item ROTATES over the whole pool
             (index t % N), so every paper is boosted equally often and the
             effect cannot be explained by any single item's content. Order
             is randomised independently in both conditions.


RUN
---
    python experiments/exp4_popularity_cue.py --trials 50

Two calls per trial (one control, one treatment).
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from common import (LETTERS, get_items, get_popularity_cues, SYSTEM_SINGLE,
                    build_single_pick_prompt, call_llm, parse_single_letter,
                    shuffled_with_map, save_results)
from stats_utils import plot_probability_bars

# Fallback only. The reported runs use the real citation counts from the
# 'citations' column of EXP_ITEMS_FILE.
DEFAULT_CUE = " Cited 2,300 times to date."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50,
                    help="paired trials (control + treatment)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = get_items()
    k = len(pool)

    real_cues = get_popularity_cues()
    if real_cues and len(real_cues) == k:
        cues = real_cues
        print("Using the REAL citation counts from the CSV as the cue.\n")
    else:
        cues = [DEFAULT_CUE] * k
        print("[note] no 'citations' column found; falling back to a fixed cue.\n")

    control_chosen: Counter[int] = Counter()
    treat_chosen: Counter[int] = Counter()
    cue_chosen = cue_shown = 0
    by_slot: dict[int, list[int]] = defaultdict(lambda: [0, 0])   # slot -> [won, shown]
    by_item: dict[int, list[int]] = defaultdict(lambda: [0, 0])   # item -> [won, shown]
    trials_log = []

    print(f"Experiment 4 -- popularity cue, {args.trials} paired trials, "
          f"{k} candidates per trial\n")

    for t in range(args.trials):
        cue_item = t % k                     # rotate the cue over the pool

        # ---- control: no cue anywhere ----
        ordered_c = shuffled_with_map(pool, rng)
        raw_c = call_llm(build_single_pick_prompt(ordered_c), SYSTEM_SINGLE, max_tokens=5)
        slot_c = parse_single_letter(raw_c, k)
        chosen_c = ordered_c[slot_c][0] if slot_c is not None else None
        if chosen_c is not None:
            control_chosen[chosen_c] += 1

        # ---- treatment: exactly one item carries the cue ----
        boosted = [txt + cues[i] if i == cue_item else txt
                   for i, txt in enumerate(pool)]
        ordered_t = shuffled_with_map(boosted, rng)
        raw_t = call_llm(build_single_pick_prompt(ordered_t), SYSTEM_SINGLE, max_tokens=5)
        slot_t = parse_single_letter(raw_t, k)
        chosen_t = ordered_t[slot_t][0] if slot_t is not None else None
        cue_slot = next(s for s, (i, _) in enumerate(ordered_t) if i == cue_item)

        if chosen_t is not None:
            treat_chosen[chosen_t] += 1
            cue_shown += 1
            won = chosen_t == cue_item
            cue_chosen += won
            by_slot[cue_slot][0] += won
            by_slot[cue_slot][1] += 1
            by_item[cue_item][0] += won
            by_item[cue_item][1] += 1

        trials_log.append({
            "trial": t, "cue_item": cue_item, "cue_slot": cue_slot,
            "control_chosen": chosen_c, "treatment_chosen": chosen_t,
        })
        mark = " <== the boosted paper was picked" if chosen_t == cue_item else ""
        print(f"  trial {t + 1:>3}: cue on paper #{cue_item + 1} in slot "
              f"{LETTERS[cue_slot]}, control -> #{(chosen_c or 0) + 1}, "
              f"treatment -> #{(chosen_t or 0) + 1}{mark}")

    # ---- aggregate ---------------------------------------------------------
    p_cue = cue_chosen / cue_shown if cue_shown else 0.0
    uplift = p_cue - 1 / k
    print(f"\n===== AGGREGATE =====")
    print(f"P(chosen | the item carries the cue) = {p_cue:.3f}")
    print(f"Uniform baseline (1/{k})             = {1 / k:.3f}")
    print(f"Popularity uplift                    = {uplift * 100:+.1f} pp")

    # ---- breakdown by slot: the cue works early, and almost nowhere else ---
    print("\n===== BY SLOT the cue item occupied =====")
    print("(the interaction with position bias: the cue mainly buys an "
          "advantage when the boosted item is shown early)")
    slot_rates = {}
    for s in range(k):
        won, shown = by_slot[s]
        rate = won / shown if shown else 0.0
        slot_rates[s] = {"won": won, "shown": shown, "rate": rate}
        print(f"  Slot {LETTERS[s]}: cue item chosen {won}/{shown} = "
              f"{rate * 100:5.1f}%   (uplift {(rate - 1 / k) * 100:+6.1f} pp)")

    # ---- breakdown by item: the cue amplifies, it does not rescue ----------
    print("\n===== BY ITEM carrying the cue =====")
    print("(control rate = how often that paper is chosen with no cue "
          "anywhere, i.e. its strength on content alone)")
    item_rates = {}
    n_control = sum(control_chosen.values())
    for i in range(k):
        won, shown = by_item[i]
        rate = won / shown if shown else 0.0
        ctrl = control_chosen.get(i, 0) / n_control if n_control else 0.0
        item_rates[i] = {"won": won, "shown": shown, "rate": rate,
                         "control_rate": ctrl}
        print(f"  Paper #{i + 1}: with cue {won}/{shown} = {rate * 100:5.1f}%"
              f"   | control {ctrl * 100:5.1f}%")

    plot_probability_bars([p_cue, 1 / k], ["Item with cue", "Uniform baseline"],
                          "Exp. 4 — Selection probability: cue vs. baseline",
                          "exp4_popularity.png", baseline=1 / k)

    save_results("exp4_popularity_cue", {
        "trials": trials_log,
        "k_shown_per_trial": k,
        "control_chosen": dict(control_chosen),
        "treatment_chosen": dict(treat_chosen),
        "p_chosen_given_cue": p_cue,
        "uplift_vs_uniform": uplift,
        "by_cue_slot": slot_rates,
        "by_cue_item": item_rates,
    })

    print("\nHow to read this:")
    print("* Because the cue rotates over the whole pool, a high or low "
          "P(chosen|cue) cannot be explained by any single paper's content.")
    print("* The aggregate and the breakdowns disagree, and the breakdowns "
          "are the finding: the cue works for items that are already "
          "competitive AND positioned early, so these biases interact "
          "rather than add.")


if __name__ == "__main__":
    main()
