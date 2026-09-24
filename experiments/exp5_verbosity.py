"""
EXPERIMENT 5 -- Verbosity.


DESIGN -- content-constant expansion with rotation
--------------------------------------------------
Per trial, exactly ONE item (rotating over the pool, index t % N) is shown in
a padded LONG form; every other item keeps its normal description. The
padding adds words, not facts: it is generic framing language that would be
true of almost any paper, so any uplift is a pure form effect. Order is
randomised as in every other experiment.

RUN
---
    python experiments/exp5_verbosity.py --trials 40

One API call per trial.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from common import (LETTERS, get_items, SYSTEM_SINGLE, build_single_pick_prompt,
                    call_llm, parse_single_letter, shuffled_with_map, save_results)
from stats_utils import plot_probability_bars

# Low-information padding: it elaborates and repeats, and states nothing that
# is not already implied. Its length (~190 words) is set to roughly double an
# extended abstract, so the manipulation is a meaningful length change rather
# than a rounding error on the prompt.
PADDING = (" This work is, generally speaking, a contribution to the broader "
           "literature on this topic; studies of this general kind have been "
           "conducted by many research groups, and it engages with the "
           "subject matter in the way one would broadly expect for a paper "
           "addressing a question of this nature. The approach taken here "
           "reflects a body of thinking that has developed over time within "
           "the wider research community, drawing on conventions for how "
           "problems of this general kind are typically framed and "
           "investigated by researchers working in adjacent areas. As with "
           "much work of this type, the contribution should be understood "
           "within the context of a field that continues to evolve, where "
           "progress is often incremental and builds upon a foundation of "
           "prior efforts spanning a range of related settings. Readers "
           "familiar with the general area will likely recognise the broad "
           "structure of the investigation, which follows patterns that are "
           "common across related studies published in comparable venues. "
           "The findings, in keeping with much research of this nature, are "
           "presented with the expectation that they will be considered "
           "alongside other contributions addressing similar or adjacent "
           "questions within the same general domain of inquiry.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = get_items()
    k = len(pool)

    long_chosen = long_shown = 0
    as_long: dict[int, list[int]] = defaultdict(lambda: [0, 0])    # item -> [won, shown]
    as_short: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    trials_log = []

    print(f"Experiment 5 -- verbosity, {args.trials} trials, "
          f"{k} candidates per trial\n")

    for t in range(args.trials):
        long_item = t % k                    # rotate the padded item
        padded = [txt + PADDING if i == long_item else txt
                  for i, txt in enumerate(pool)]
        ordered = shuffled_with_map(padded, rng)
        raw = call_llm(build_single_pick_prompt(ordered), SYSTEM_SINGLE, max_tokens=5)
        slot = parse_single_letter(raw, k)
        if slot is None:
            print(f"  trial {t + 1:>3}: unparseable, skipped")
            continue

        chosen = ordered[slot][0]
        long_shown += 1
        long_chosen += chosen == long_item
        for i in range(k):
            bucket = as_long if i == long_item else as_short
            bucket[i][0] += chosen == i
            bucket[i][1] += 1

        long_slot = next(s for s, (i, _) in enumerate(ordered) if i == long_item)
        trials_log.append({"trial": t, "long_item": long_item,
                           "long_slot": long_slot, "chosen": chosen})
        mark = " <== the verbose paper was picked" if chosen == long_item else ""
        print(f"  trial {t + 1:>3}: verbose = paper #{long_item + 1} in slot "
              f"{LETTERS[long_slot]}, chosen = #{chosen + 1}{mark}")

    # ---- aggregate ---------------------------------------------------------
    p_long = long_chosen / long_shown if long_shown else 0.0
    p_short_avg = (1 - p_long) / (k - 1) if k > 1 else 0.0
    uplift = p_long - 1 / k
    print("\n===== AGGREGATE =====")
    print(f"P(chosen | shown in LONG form) = {p_long:.3f}")
    print(f"Uniform baseline (1/{k})        = {1 / k:.3f}")
    print(f"Verbosity uplift               = {uplift * 100:+.1f} pp")

    # ---- per item: the effect is graded, not uniform -----------------------
    print("\n===== BY ITEM (pick rate when padded vs when concise) =====")
    per_item = {}
    for i in range(k):
        lw, ls = as_long[i]
        sw, ss = as_short[i]
        lr = lw / ls if ls else 0.0
        sr = sw / ss if ss else 0.0
        per_item[i] = {"long_won": lw, "long_shown": ls, "long_rate": lr,
                       "short_won": sw, "short_shown": ss, "short_rate": sr}
        print(f"  Paper #{i + 1}: padded {lw}/{ls} = {lr * 100:5.1f}%   "
              f"| concise {sw}/{ss} = {sr * 100:5.1f}%   "
              f"| change {(lr - sr) * 100:+6.1f} pp")

    plot_probability_bars([p_long, p_short_avg],
                          ["Verbose item", "Avg. concise item"],
                          "Exp. 5 — Selection probability: verbose vs concise form",
                          "exp5_verbosity.png", baseline=1 / k)

    save_results("exp5_verbosity", {
        "trials": trials_log,
        "k_shown_per_trial": k,
        "p_chosen_given_long": p_long,
        "p_avg_concise_item": p_short_avg,
        "uplift_vs_uniform": uplift,
        "by_item": per_item,
    })

    print("\nHow to read this:")
    print("* The padding adds words but no facts, so any uplift is a pure "
          "form effect.")
    print("* Compare the per-item column: papers at zero stay at zero when "
          "padded, while already-competitive papers gain. Length is a "
          "tie-breaker, not a rescue.")


if __name__ == "__main__":
    main()
