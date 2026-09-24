"""
EXPERIMENT 8, part 1 -- Mitigation in ranking: the API-costing conditions.

RUN
---
    python experiments/exp8_mitigation_ranking.py --trials 30
    python experiments/exp8_ate_analysis.py          # then this

Cost per trial = 2 x (1 + K + (K-1)) calls.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (get_items, SYSTEM_SINGLE, SYSTEM_RANKING,
                    build_single_pick_prompt, build_ranking_prompt,
                    parse_single_letter, parse_ranking, call_llm,
                    shuffled_with_map, save_results, PROVIDER, MODEL)
from stats_utils import kendall_tau


# ---------------------------------------------------------------------------
# Strategies. Each takes an ORDERED candidate list of (item_id, text) and
# returns a ranking as a list of item ids, best first, or None if the
# model's output could not be parsed.
# ---------------------------------------------------------------------------

def strat_standard(ordered: list[tuple[int, str]]) -> list[int] | None:
    """Baseline: ask for the whole ranking in one call."""
    raw = call_llm(build_ranking_prompt(ordered), SYSTEM_RANKING, max_tokens=40)
    slots = parse_ranking(raw, len(ordered))
    return [ordered[s][0] for s in slots] if slots is not None else None


def borda_aggregate(rankings: list[list[int]], items: list[int]) -> list[int]:
    """Borda count: an item in position p of a K-item ranking scores K-1-p.
    Higher total is better; ties broken by item id for determinism."""
    score: dict[int, int] = defaultdict(int)
    k = len(items)
    for r in rankings:
        for pos, item in enumerate(r):
            score[item] += k - 1 - pos
    return sorted(items, key=lambda it: (-score[it], it))


def collect_runs(orders: list[list[tuple[int, str]]]) -> list[dict] | None:
    """Run the standard ranking call once per candidate ordering and keep the
    raw record: the output ranking plus the input slot each item occupied."""
    runs = []
    for ordering in orders:
        r = strat_standard(ordering)
        if r is None:
            continue
        runs.append({
            "ranking": r,
            "input_slot": {item: slot for slot, (item, _) in enumerate(ordering)},
        })
    return runs or None


def strat_psc(ordered: list[tuple[int, str]], rng: random.Random):
    """PSC / Borda (Tang et al. 2024): K random reshuffles of the same
    candidate set, aggregated by Borda count. Returns (ranking, raw_runs)."""
    k = len(ordered)
    orders = []
    for _ in range(k):
        shuffled = ordered[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
    runs = collect_runs(orders)
    if runs is None:
        return None, None
    items = [it for it, _ in ordered]
    return borda_aggregate([r["ranking"] for r in runs], items), runs


def strat_rise(ordered: list[tuple[int, str]]) -> list[int] | None:
    """RISE@1 (Bito et al. 2025): ask for ONE item, append it to the ranking,
    remove it from the candidate list, repeat."""
    remaining = ordered[:]
    ranking: list[int] = []
    while len(remaining) > 1:
        raw = call_llm(build_single_pick_prompt(remaining), SYSTEM_SINGLE, max_tokens=5)
        slot = parse_single_letter(raw, len(remaining))
        if slot is None:
            return None
        ranking.append(remaining[slot][0])
        remaining.pop(slot)
    ranking.append(remaining[0][0])      # the last item needs no call
    return ranking


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = get_items()
    k = len(pool)

    print(f"Experiment 8 (part 1) -- mitigation in ranking, {args.trials} "
          f"trials, {k} candidates per trial")
    print(f"Provider: {PROVIDER} | model: {MODEL}")
    print(f"Cost: 2 x (1 + K + (K-1)) = {2 * (1 + k + k - 1)} calls per trial")
    print("ATE is evaluated offline from the saved runs; run "
          "exp8_ate_analysis.py afterwards.\n")

    strategies = ["standard", "psc", "rise"]
    pcs: dict[str, list[float]] = {s: [] for s in strategies}
    skipped: dict[str, int] = {s: 0 for s in strategies}
    trials_log = []
    raw_runs: dict[str, list] = {"psc": []}

    for t in range(args.trials):
        base = shuffled_with_map(pool, rng)    # a shuffled candidate list
        reversed_base = base[::-1]             # the same items, opposite order

        row = {"trial": t + 1}
        for name in strategies:
            if name == "standard":
                r1, r2 = strat_standard(base), strat_standard(reversed_base)
            elif name == "psc":
                (r1, runs1) = strat_psc(base, rng)
                (r2, runs2) = strat_psc(reversed_base, rng)
                raw_runs["psc"].append({"trial": t + 1, "fwd": runs1, "rev": runs2})
            else:
                r1, r2 = strat_rise(base), strat_rise(reversed_base)

            if r1 is None or r2 is None:
                skipped[name] += 1
                row[name] = None
                continue
            tau = kendall_tau(r1, r2)
            pcs[name].append(tau)
            row[name] = round(tau, 4)

        trials_log.append(row)
        print(f"  trial {t + 1:>3}: " + "  ".join(
            f"{n}={row[n] if row[n] is not None else 'skip':>6}" for n in strategies))

    # ---- summary -----------------------------------------------------------
    labels = {"standard": "1. Standard (baseline)", "psc": "2. PSC / Borda",
              "rise": "3. RISE@1"}
    cost = {"standard": 1, "psc": k, "rise": k}
    print("\n===== Positional Consistency (Kendall's tau between a shuffled "
          "list and its exact reverse) =====")
    print("Higher = more consistent = less position bias. 1.0 = fully "
          "order-invariant.\n")
    print(f"{'Strategy':<24}{'Mean PC':<10}{'Std':<9}{'Calls':<8}{'n valid'}")
    summary = {}
    for name in strategies:
        vals = pcs[name]
        if not vals:
            print(f"{labels[name]:<24}{'--':<10}{'--':<9}{cost[name]:<8}0")
            continue
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary[name] = {"mean_pc": m, "std_pc": sd, "n": len(vals),
                         "calls_per_ranking": cost[name]}
        print(f"{labels[name]:<24}{m:<10.3f}{sd:<9.3f}{cost[name]:<8}{len(vals)}")

    if any(skipped.values()):
        print("\nSkipped (unparseable): " +
              ", ".join(f"{n}={c}" for n, c in skipped.items() if c))

    save_results("exp8_mitigation_ranking", {
        "trials": args.trials, "k": k, "summary": summary,
        "skipped": skipped, "per_trial": trials_log, "raw_runs": raw_runs,
    })

    print("\nHow to read this:")
    print("* Any drop below 1.0 is order sensitivity: the two inputs contain "
          "the same items and differ only in direction.")
    print("* Conditions 2 and 3 are cost-matched, so the gap between them is "
          "about strategy and not budget; condition 1 is the reference.")
    print("* Run exp8_ate_analysis.py to add ATE and produce the figures.")


if __name__ == "__main__":
    main()
