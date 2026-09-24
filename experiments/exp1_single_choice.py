"""
EXPERIMENT 1 -- Single choice.

DESIGN
------
Show all six candidates, ask for one, reshuffle every trial. Each trial is a
fresh permutation plus one LLM call; the model must answer with a single
option letter.

Two things are recorded:
  * which SLOT was chosen  -> position analysis (the core metric)
  * which ITEM was chosen  -> content analysis

RUN
---
    python experiments/exp1_single_choice.py --trials 30

One API call per trial.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from common import (LETTERS, get_items, SYSTEM_SINGLE, build_single_pick_prompt,
                    call_llm, parse_single_letter, shuffled_with_map, save_results)
from stats_utils import (plot_selection_rates, print_summary,
                         chi_square_uniform, shannon_entropy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = get_items()
    k = len(pool)

    slot_counts = [0] * k                  # position analysis
    item_chosen: Counter[int] = Counter()  # content analysis
    trials_log = []

    print(f"Experiment 1 -- single choice, {args.trials} trials, "
          f"{k} candidates per trial\n")

    for t in range(args.trials):
        ordered = shuffled_with_map(pool, rng)
        raw = call_llm(build_single_pick_prompt(ordered), SYSTEM_SINGLE, max_tokens=5)
        slot = parse_single_letter(raw, k)
        if slot is None:
            print(f"  trial {t + 1:>3}: unparseable answer ({raw!r}), skipped")
            continue
        item = ordered[slot][0]
        slot_counts[slot] += 1
        item_chosen[item] += 1
        trials_log.append({
            "trial": t,
            "order": [i for i, _ in ordered],   # item id per slot
            "chosen_slot": slot,
            "chosen_item": item,
            "raw": raw,
        })
        print(f"  trial {t + 1:>3}: chose slot {LETTERS[slot]} (paper #{item + 1})")

    # --- position analysis --------------------------------------------------
    slot_labels = [f"Slot {LETTERS[i]}" for i in range(k)]
    print_summary("POSITION analysis (which slot was picked)", slot_counts, slot_labels)
    plot_selection_rates(slot_counts, slot_labels,
                         "Exp. 1 — Selection rate by prompt position",
                         "exp1_position_rates.png", baseline=1 / k)
    chi2_pos, p_pos = chi_square_uniform(slot_counts)

    # --- content analysis ---------------------------------------------------
    
    item_counts = [item_chosen.get(i, 0) for i in range(k)]
    item_labels = [f"Paper {i + 1}" for i in range(k)]
    print_summary("CONTENT analysis (which paper was picked)", item_counts, item_labels)
    plot_selection_rates(item_counts, item_labels,
                         "Exp. 1 — Selection rate by item identity",
                         "exp1_item_rates.png", baseline=1 / k,
                         xlabel="Candidate paper")
    chi2_item, p_item = chi_square_uniform(item_counts)

    save_results("exp1_single_choice", {
        "trials": trials_log,
        "k_shown_per_trial": k,
        "slot_counts": slot_counts,
        "chi2_position": chi2_pos, "p_position": p_pos,
        "entropy_position": shannon_entropy(slot_counts),
        "item_counts": item_counts,
        "chi2_item": chi2_item, "p_item": p_item,
        "entropy_item": shannon_entropy(item_counts),
    })

    print("\nHow to read this:")
    print("* p_position < 0.05 -> significant position bias: some slots are "
          "chosen more than the 1/K uniform baseline.")
    print("* p_item < 0.05 -> a content preference is present as well. Both "
          "effects can be real at once, and in this format they are "
          "entangled; Experiments 2 and 3 separate them.")


if __name__ == "__main__":
    main()
