"""
EXPERIMENT 7 -- Mitigation in single choice.

RUN
---
    python experiments/exp7_mitigation_single_choice.py --decisions 35 --k 5
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (get_items, SYSTEM_SINGLE, build_single_pick_prompt,
                    call_llm, parse_single_letter, shuffled_with_map, save_results)
from stats_utils import (shannon_entropy, PLOTS_DIR, C_GRAY, C_RED,
                         C_DBLUE, C_TEAL, C_NAVY, C_GRID)

SYSTEM_DEBIAS = (
    "You are a neutral recommendation assistant. The order in which options "
    "are listed and their letter labels are RANDOM and carry no information; "
    "judge options purely on their content, ignoring their position in the "
    "list. Answer with exactly one capital letter and nothing else."
)


def decide_shuffled(subset: list[tuple[int, str]], rng: random.Random,
                    system: str) -> int | None:
    """One reshuffled single-pick call over a fixed candidate set.
    Returns the chosen item's id."""
    ordered = list(subset)
    rng.shuffle(ordered)
    raw = call_llm(build_single_pick_prompt(ordered), system, max_tokens=5)
    slot = parse_single_letter(raw, len(ordered))
    return ordered[slot][0] if slot is not None else None


def decide_fixed(ordered: list[tuple[int, str]], system: str) -> int | None:
    """One single-pick call with the candidates in the EXACT given order.
    Used by ATE's probe, which needs one specific order and its reverse."""
    raw = call_llm(build_single_pick_prompt(ordered), system, max_tokens=5)
    slot = parse_single_letter(raw, len(ordered))
    return ordered[slot][0] if slot is not None else None


def psc_decision(subset: list[tuple[int, str]], rng: random.Random,
                 k: int) -> int | None:
    """PSC adapted to single choice: K reshuffled calls over the same
    candidate set, aggregated by majority vote over item identity."""
    votes: Counter[int] = Counter()
    for _ in range(k):
        item = decide_shuffled(subset, rng, SYSTEM_SINGLE)
        if item is not None:
            votes[item] += 1
    return votes.most_common(1)[0][0] if votes else None


def ate_decision(subset: list[tuple[int, str]], rng: random.Random,
                 k: int) -> tuple[int | None, int]:
    """ATE, single-choice form. "Agree" means both probe calls returned the
    same item. Returns (chosen item id, calls actually spent)."""
    base = list(subset)
    rng.shuffle(base)

    votes: Counter[int] = Counter()
    d_fwd = decide_fixed(base, SYSTEM_SINGLE)
    d_rev = decide_fixed(base[::-1], SYSTEM_SINGLE)
    for d in (d_fwd, d_rev):
        if d is not None:
            votes[d] += 1

    if d_fwd is not None and d_fwd == d_rev:
        return d_fwd, 2                        # not contested: stop here

    # Contested (or one side unparseable): spend the full K-call budget,
    # of which the two probe calls are already paid for.
    for _ in range(max(k - 2, 0)):
        item = decide_shuffled(subset, rng, SYSTEM_SINGLE)
        if item is not None:
            votes[item] += 1
    return (votes.most_common(1)[0][0] if votes else None), k


def summarise(name: str, decisions: list[int], calls_per_decision: float,
              escalated: int | None = None, total: int | None = None) -> dict:
    counts = Counter(decisions)
    ent = shannon_entropy(list(counts.values())) if counts else 0.0
    modal = max(counts.values()) / len(decisions) if decisions else 0.0
    print(f"\n--- {name} ---")
    print(f"  decisions by paper: {[(f'#{i + 1}', c) for i, c in counts.most_common()]}")
    line = (f"  entropy = {ent:.3f} bits | modal agreement = {modal * 100:.0f}% "
            f"| cost = {calls_per_decision:.2f} calls/decision")
    if escalated is not None and total:
        line += f" | escalated {escalated}/{total} ({escalated / total * 100:.0f}%)"
    print(line)
    return {"counts": dict(counts), "entropy": ent, "modal_agreement": modal,
            "calls_per_decision": calls_per_decision,
            "escalated": escalated, "n_decisions": total}


def plot_cost_vs_stability(summaries: dict[str, dict]) -> None:
    """Two panels: what each strategy buys, and what it costs to buy it.

    This is the figure as presented in the thesis defence; the marker shapes,
    colours and annotations are fixed so that the figure in the write-up and
    the figure in the talk are the same figure.
    """
    names = list(summaries)
    ent = [summaries[n]["entropy"] for n in names]
    calls = [summaries[n]["calls_per_decision"] for n in names]
    modal = [summaries[n]["modal_agreement"] * 100 for n in names]
    colors = [C_GRAY, C_RED, C_DBLUE, C_TEAL]
    # (marker, size, label offset in points, horizontal alignment)
    # order matches `names`: baseline, debias prompt, PSC, ATE
    styles = [("o", 140, (16, -14), "left"),
              ("s", 125, (16, 9), "left"),
              ("D", 135, (-13, -17), "right"),
              ("*", 460, (-17, 0), "right")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.6))

    # --- top: entropy against cost --------------------------------------
    for n, x, y, c, (mk, sz, off, ha) in zip(names, calls, ent, colors, styles):
        ax1.scatter(x, y, marker=mk, s=sz, color=c, zorder=4,
                    linewidths=0)
        ax1.annotate(n, (x, y), textcoords="offset points", xytext=off,
                     fontsize=10, color=c, ha=ha, va="center",
                     fontweight="bold" if mk == "*" else "normal")
    # the cost gap between the proposed method and the strategy it matches
    if len(calls) == 4 and calls[3] < calls[2]:
        saving = (1 - calls[3] / calls[2]) * 100
        y_ann = min(ent) + 0.36 * (max(ent) - min(ent))
        ax1.annotate("", xy=(calls[3], y_ann), xytext=(calls[2], y_ann),
                     arrowprops=dict(arrowstyle="<->", color=C_TEAL, lw=1.3))
        ax1.text((calls[2] + calls[3]) / 2, y_ann + 0.012,
                 f"-{saving:.1f}%\ncost", ha="center", va="bottom",
                 fontsize=8.5, color=C_TEAL, linespacing=1.2)

    ax1.set_xlim(min(calls) - 0.45, max(calls) + 0.55)
    ax1.set_ylim(min(ent) - 0.06, max(ent) + 0.06)
    ax1.set_xlabel("API calls per decision", fontsize=10)
    ax1.set_ylabel("Entropy of final decisions (bits)", fontsize=10)
    ax1.set_title("Stability bought per unit of cost", fontsize=12,
                  fontweight="bold", color=C_NAVY)
    ax1.text(0.012, 0.03, "lower = more stable", transform=ax1.transAxes,
             fontsize=8.5, style="italic", color="#7a7a7a")
    ax1.grid(color=C_GRID, linewidth=0.9)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # --- bottom: modal agreement ----------------------------------------
    xs = range(len(names))
    ax2.bar(xs, modal, color=colors[:len(names)], width=0.62)
    for i, (v, c) in enumerate(zip(modal, colors)):
        ax2.text(i, v + 1.4, f"{v:.0f}%", ha="center", fontsize=11,
                 fontweight="bold", color=c)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=9)
    ax2.set_ylim(0, max(modal) * 1.14)
    ax2.set_ylabel("Modal agreement (%)", fontsize=10)
    ax2.set_title("How often the same item is chosen", fontsize=12,
                  fontweight="bold", color=C_NAVY)
    ax2.grid(axis="y", color=C_GRID, linewidth=0.9)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    out = PLOTS_DIR / "exp7_mitigation_single_choice.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nSaved plot -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=35,
                    help="independent final decisions per condition (M)")
    ap.add_argument("--k", type=int, default=5,
                    help="calls aggregated per PSC decision (K)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = get_items()
    M, K = args.decisions, args.k

    print(f"Experiment 7 -- mitigation in single choice, M={M} decisions, "
          f"K={K}, {len(pool)} candidates per trial\n")

    baseline, debias, psc, ate = [], [], [], []
    ate_calls: list[int] = []
    ate_escalated = 0

    for m in range(M):
        subset = shuffled_with_map(pool, rng)
        print(f"decision {m + 1}/{M}: baseline...", end=" ", flush=True)
        d1 = decide_shuffled(subset, rng, SYSTEM_SINGLE)
        print(f"#{(d1 or 0) + 1} | debias...", end=" ", flush=True)
        d2 = decide_shuffled(subset, rng, SYSTEM_DEBIAS)
        print(f"#{(d2 or 0) + 1} | PSC({K})...", end=" ", flush=True)
        d3 = psc_decision(subset, rng, K)
        print(f"#{(d3 or 0) + 1} | ATE...", end=" ", flush=True)
        d4, calls4 = ate_decision(subset, rng, K)
        print(f"#{(d4 or 0) + 1} ({calls4} calls)")

        for value, bucket in ((d1, baseline), (d2, debias), (d3, psc), (d4, ate)):
            if value is not None:
                bucket.append(value)
        ate_calls.append(calls4)
        ate_escalated += calls4 > 2

    ate_avg = sum(ate_calls) / len(ate_calls) if ate_calls else 0.0
    summaries = {
        "Baseline": summarise("Baseline (1 call)", baseline, 1.0),
        "Debias prompt": summarise("Debias prompt (1 call)", debias, 1.0),
        f"PSC (K={K})": summarise(f"PSC, K={K} (majority vote)", psc, float(K)),
        "ATE (proposed)": summarise(f"ATE (proposed), K={K}", ate, ate_avg,
                                    escalated=ate_escalated, total=M),
    }
    plot_cost_vs_stability(summaries)

    print("\n===== SUMMARY =====")
    print(f"{'Strategy':<18}{'Entropy':<11}{'Modal agr.':<13}{'Calls'}")
    for name, s in summaries.items():
        print(f"{name:<18}{s['entropy']:<11.3f}"
              f"{s['modal_agreement'] * 100:<13.0f}{s['calls_per_decision']:.2f}")
    if ate_avg:
        print(f"\nATE cost saving vs PSC: {(1 - ate_avg / K) * 100:.1f}% "
              f"(escalated on {ate_escalated} of {M} decisions)")

    save_results("exp7_mitigation_single_choice", {
        "baseline": summaries["Baseline"],
        "debias_prompt": summaries["Debias prompt"],
        "psc": summaries[f"PSC (K={K})"],
        "ate": summaries["ATE (proposed)"],
        "raw_decisions": {"baseline": baseline, "debias": debias,
                          "psc": psc, "ate": ate},
        "ate_calls_per_decision": ate_calls,
        "K": K, "M": M, "k_shown_per_trial": len(pool),
    })

    print("\nHow to read this:")
    print("* Aggregation beats prompt-only debiasing. A debias prompt that "
          "scores worse than the baseline replicates the finding that "
          "instruction-only mitigation is weak.")
    print("* ATE targets a different axis from 1-3: not more stability, but "
          "the same stability at lower average cost. Compare it against PSC "
          "at its variable cost, not against the fixed K.")
    print("* A high escalation rate means most decisions here really were "
          "contested, so the saving in this format is small.")
    print("* All four conditions converge on the item already dominant on "
          "content: what is demonstrated is stability, not correctness.")


if __name__ == "__main__":
    main()
