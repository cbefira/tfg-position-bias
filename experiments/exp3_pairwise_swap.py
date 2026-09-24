"""
EXPERIMENT 3 -- Pairwise swap.

DESIGN
------
Every unordered pair of the six papers is tested in both presentations
(X first, then Y first), repeated for --rounds sweeps.


RUN
---
    python experiments/exp3_pairwise_swap.py --rounds 3

With six papers: 15 pairs x 2 presentations x --rounds calls.
"""

from __future__ import annotations

import argparse
import itertools

from common import (get_items, SCENARIO, SYSTEM_SINGLE, call_llm,
                    parse_single_letter, save_results)
from stats_utils import plot_grouped_rates


def build_pair_prompt(first: str, second: str) -> str:
    return (f"{SCENARIO}\n\nCandidate options:\n\n(A) {first}\n(B) {second}\n\n"
            "Which option do you recommend? Reply with only the single letter.")


def ask_pair(x: str, y: str) -> int | None:
    """Present x as A and y as B. Returns 0 if the model chose slot A
    (i.e. x), 1 if it chose slot B (i.e. y), None if unparseable."""
    raw = call_llm(build_pair_prompt(x, y), SYSTEM_SINGLE, max_tokens=5)
    return parse_single_letter(raw, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3,
                    help="how many times to sweep all pairs")
    args = ap.parse_args()

    # No RNG here: the pair list is exhaustive and its order is fixed, so
    # the design is deterministic and needs no seed.
    pool = get_items()
    base_pairs = list(itertools.combinations(range(len(pool)), 2))
    pair_list = base_pairs * args.rounds

    print(f"Experiment 3 -- pairwise swap, {len(base_pairs)} pairs x "
          f"{args.rounds} rounds ({len(pair_list) * 2} calls)\n")

    consistent = inconsistent = 0
    primacy_reversals = 0     # flips where BOTH presentations picked slot A
    first_slot_wins = 0       # call-level count within flipped pairs
    a_answers = total_answers = 0
    trials_log = []

    for n, (i, j) in enumerate(pair_list, 1):
        v1 = ask_pair(pool[i], pool[j])     # presentation 1: i first
        v2 = ask_pair(pool[j], pool[i])     # presentation 2: j first
        if v1 is None or v2 is None:
            print(f"  pair {n:>4}: unparseable, skipped")
            continue
        total_answers += 2
        a_answers += (v1 == 0) + (v2 == 0)

        winner1 = i if v1 == 0 else j
        winner2 = j if v2 == 0 else i
        is_consistent = winner1 == winner2
        if is_consistent:
            consistent += 1
        else:
            inconsistent += 1
            # A flip means the model followed a slot rather than an item.
            first_slot_wins += (v1 == 0) + (v2 == 0)
            if v1 == 0 and v2 == 0:
                primacy_reversals += 1

        trials_log.append({
            "pair_index": n, "item_i": i, "item_j": j,
            "winner_pres1": winner1, "winner_pres2": winner2,
            "consistent": is_consistent,
        })
        print(f"  pair {n:>4}: paper #{i + 1} vs #{j + 1} -> "
              f"{'consistent' if is_consistent else 'FLIPPED'}")

    judged = consistent + inconsistent
    cons_rate = consistent / judged if judged else 0.0
    a_rate = a_answers / total_answers if total_answers else 0.0
    primacy = first_slot_wins / (2 * inconsistent) if inconsistent else float("nan")

    print(f"\nConsistency under swap: {consistent}/{judged} ({cons_rate * 100:.1f}%)")
    print(f"Flipped by swap:        {inconsistent}/{judged} "
          f"({(1 - cons_rate) * 100:.1f}%)")
    if inconsistent:
        print(f"Reversals favouring the item shown FIRST: "
              f"{primacy_reversals}/{inconsistent}")
        print(f"  (call-level: the winner sat in the first slot "
              f"{primacy * 100:.1f}% of the time; >50% primacy, <50% recency)")
    print(f"Share of 'A' answers overall: {a_rate * 100:.1f}% "
          f"(50% = no first-slot or label preference)")

    plot_grouped_rates({"Observed": [consistent, inconsistent]},
                       ["Consistent", "Flipped by swap"],
                       "Exp. 3 — Pairwise verdicts under order swap",
                       "exp3_pairwise_swap.png")

    save_results("exp3_pairwise_swap", {
        "trials": trials_log,
        "pool_size": len(pool),
        "rounds": args.rounds,
        "consistent": consistent,
        "inconsistent": inconsistent,
        "consistency_rate": cons_rate,
        "primacy_reversals": primacy_reversals,
        "share_A_answers": a_rate,
        "first_slot_win_rate_in_flips": primacy,
    })

    print("\nHow to read this:")
    print("* Every flipped pair is an unambiguous order-driven decision: the "
          "two presentations carry identical information.")
    print("* The direction of the reversals establishes not only that the "
          "effect exists but which way it points.")
    print("* A high share of 'A' answers may combine primacy with a "
          "preference for the token 'A' itself; both explanations are "
          "reported rather than assumed apart.")


if __name__ == "__main__":
    main()
