# Position Bias in LLM-Based Recommender Systems — experiment suite

Code for the bachelor's thesis *Position Bias in LLM-Based Recommender
Systems* (Carmen Beferull Iranzo, FAU Erlangen-Nürnberg).

Eight experiments plus a cross-experiment comparison. Every number the
scripts report comes from live LLM API calls; nothing is simulated. Raw
trial logs, computed metrics and run metadata are written to `results/`,
and the figures to `plots/`.

## The protocol

All eight experiments share one procedure, and the reshuffle is the only
thing that varies between trials:

```
item pool (6 real papers)
  → random reshuffle
  → one of three prompt formats   (single choice · listwise ranking · pairwise swap)
  → one LLM call
  → record the decision
repeated 30–60 times
```

The scenario is the same throughout: choosing which paper to cite as the
primary reference for a claim.

## The experiments

| # | Script | What it measures | Calls |
|---|--------|------------------|-------|
| 1 | `experiments/exp1_single_choice.py` | Position bias when six candidates are shown and one is asked for. Reports the slot distribution and the item distribution separately | 30 |
| 2 | `experiments/exp2_listwise_ranking.py` | Position bias across the whole ordering: mean output rank per input slot, and Kendall's τ across reshuffles | 60 |
| 3 | `experiments/exp3_pairwise_swap.py` | Each pair judged in both orders. The content confound is removed by construction, so every flipped verdict is one order-driven decision | 90 |
| 4 | `experiments/exp4_popularity_cue.py` | A real citation count attached to one rotating item. Breaks the effect down by the slot the cue item occupied and by which paper carried it | 100 |
| 5 | `experiments/exp5_verbosity.py` | One rotating item padded with text that adds no facts. Per-item breakdown shows the effect is graded | 40 |
| 6 | `experiments/exp6_model_comparison.py` | The same protocol across three Claude capability tiers | 30 × 3 |
| 7 | `experiments/exp7_mitigation_single_choice.py` | Baseline vs debias prompt vs PSC vs ATE, on stability and on cost | ~270 |
| 8 | `experiments/exp8_mitigation_ranking.py` → `experiments/exp8_ate_analysis.py` | Standard vs RISE vs PSC/Borda on positional consistency; then ATE, evaluated offline from the saved runs across the whole threshold range | 720 + 0 |
| — | `compare_biases.py` | Position, popularity and verbosity on a single scale | 0 |

Experiment 8 runs in two parts. Part 1 makes the API calls and saves each
individual run together with the input slot every item occupied; part 2
reconstructs ATE from that saved data and produces both figures for the
experiment, at no additional API cost.

## Setup

```bash
pip install -r requirements.txt

# Experiments 1–5, 7 and 8: Llama 3.3 70B
export EXP_PROVIDER=together
export TOGETHER_API_KEY="..."
export EXP_ITEMS_FILE=papers.csv

# Experiment 6: the three Claude capability tiers
export ANTHROPIC_API_KEY="sk-ant-..."
```

`EXP_ITEMS_FILE` points at the candidate pool (`papers.csv`, columns
`title,abstract,year,citations`). The citation counts in that file are what
Experiment 4 uses as its popularity cue. Leaving the variable unset falls
back to the identical pool hard-coded in `common.py`.

Other variables: `EXP_MODEL` overrides the model id, `EXP_TEMPERATURE`
defaults to 0.7 (non-zero is required — at temperature 0 each exact prompt
string yields one deterministic answer and the position analysis
degenerates), `EXP_MIN_INTERVAL` sets the pacing between calls.

Llama 3.3 70B is served here through Together AI; Groq, which served it
originally, retired the model id on 2026-08-16. Both point at the same
weights, and `EXP_PROVIDER=groq` still works if the id returns.

## Reproducing the reported runs

Run from the project root, in this order:

```bash
python experiments/exp1_single_choice.py            --trials 30
python experiments/exp2_listwise_ranking.py         --trials 60
python experiments/exp3_pairwise_swap.py            --rounds 3
python experiments/exp4_popularity_cue.py           --trials 50
python experiments/exp5_verbosity.py                --trials 40
python experiments/exp6_model_comparison.py         --trials 30
python experiments/exp7_mitigation_single_choice.py --decisions 35 --k 5
python experiments/exp8_mitigation_ranking.py       --trials 30
python experiments/exp8_ate_analysis.py
python compare_biases.py
```

These are the defaults, so the flags are shown only to make the sample
sizes explicit. Permutations use a seeded RNG (`--seed`), so the
presentation orders are reproducible; the model's responses are stochastic,
which is inherent to the object of study and is discussed in the
limitations.

`results/` and `plots/` in this repository hold the outputs of exactly these
runs, produced with Llama 3.3 70B at temperature 0.7 (Experiment 6 with the
three Claude tiers). Each JSON records the provider, model, temperature and
timestamp under `_meta`.

## Metrics

Each metric is taken from a published instrument rather than invented here,
so the results are directly comparable to the literature:

| Metric | What it captures | Source |
|---|---|---|
| χ² against uniform | Whether the distribution of picks across slots departs from chance | Pezeshkpour & Hruschka |
| Kendall's τ | Agreement between two rankings of the same items | Bito et al.; Fang et al. |
| Consistency under swap | Whether a verdict survives exchanging the order of a pair | MT-Bench (Zheng et al.) |
| Shannon entropy | Stability of the final decision across reshuffles | Standard information measure |
| Positional consistency | τ between a ranking and the ranking of its exact reverse | Bito et al. (2025) |

Implementations are in `stats_utils.py`.

## Mitigation strategies and their origin

| Strategy | Mechanism | Provenance |
|---|---|---|
| Debias prompt | Instruct the model that the order is random and should be ignored | Standard weak baseline |
| PSC — ranking | K reshuffled rankings aggregated by Borda count | Tang et al. (2024), as published |
| RISE | Rebuild the ranking through repeated single selection | Bito et al. (2025), as published |
| PSC — single choice | K reshuffled picks aggregated by majority vote | Adapted in this thesis |
| ATE | Two-call probe; escalate to the full K runs only on disagreement | Proposed in this thesis |

The original PSC aggregates rankings with Borda count. A single-pick task
has no ranking to aggregate, so the single-choice form is an adaptation
rather than a citation, and the code labels it as such.

## Layout

```
common.py                      pool, LLM client, prompt templates, result writer
stats_utils.py                 metrics and plotting helpers
papers.csv                     the six candidate papers and their citation counts
compare_biases.py              cross-experiment comparison
tfg_bias_experiments.ipynb     notebook runner: one section per experiment
experiments/                   one script per experiment
results/*.json                 raw trial logs, metrics and run metadata
plots/*.png                    figures at 200 dpi
```

Scripts import from the project root, so run them from there or add the root
to `PYTHONPATH`. The notebook calls the same `main()` functions with the
sample sizes above and shows each figure inline; it holds no logic of its
own, so the scripts remain the source of truth. API keys are prompted with
`getpass` and never written into the notebook file.
