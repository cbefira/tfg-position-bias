"""
common.py -- Shared infrastructure for all eight experiments.

Bachelor's thesis: "Position Bias in LLM-Based Recommender Systems"
(Carmen Beferull Iranzo, FAU Erlangen-Nurnberg, Universitat Politecnica de Valencia).


THE PROTOCOL (identical across experiments)
-------------------------------------------
    item pool (6 real papers)
      -> random reshuffle              <- the only thing that varies
      -> one of three prompt formats   (single choice / ranking / pairwise)
      -> one LLM call
      -> record the decision
    repeated 30-60 times.

CONFIGURATION (environment variables)
-------------------------------------
    EXP_PROVIDER     together (default) | groq | anthropic
    EXP_MODEL        override the model id for the chosen provider
    EXP_TEMPERATURE  default 0.7 -- non-zero is required: at temperature 0
                     each exact prompt string yields one deterministic
                     answer and the position analysis degenerates
    EXP_MIN_INTERVAL seconds between calls (rate-limit pacing)
    EXP_ITEMS_FILE   CSV with the candidate papers (see papers.csv);
                     unset falls back to the built-in PAPER_ITEMS

    # Experiments 1-5, 7, 8 (Llama 3.3 70B):
    export EXP_PROVIDER=together
    export TOGETHER_API_KEY="..."
    export EXP_ITEMS_FILE=papers.csv

    # Experiment 6 (the three Claude capability tiers):
    export ANTHROPIC_API_KEY="sk-ant-..."

"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from pathlib import Path

import requests

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PROVIDER = os.environ.get("EXP_PROVIDER", "together").lower()

_DEFAULT_MODELS = {
    # Llama 3.3 70B is the model under test in Experiments 1-5, 7 and 8.
    # Together AI is used because Groq, which served it originally, retired
    # the model id on 2026-08-16; both entries point at the same weights.
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-sonnet-5",
}
MODEL = os.environ.get("EXP_MODEL", _DEFAULT_MODELS.get(PROVIDER, "claude-sonnet-5"))
TEMPERATURE = float(os.environ.get("EXP_TEMPERATURE", "0.7"))

# Seconds between calls, per provider, to stay inside rate limits.
_DEFAULT_INTERVAL = {"together": 0.0, "groq": 2.5, "anthropic": 0.0}
MIN_INTERVAL = float(os.environ.get("EXP_MIN_INTERVAL",
                                    _DEFAULT_INTERVAL.get(PROVIDER, 2.0)))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

LETTERS = list("ABCDEFGH")
MAX_ITEMS = len(LETTERS)

# Model catalog for Experiment 6. These are three CAPABILITY TIERS of the
# same vendor and generation (small / medium / large), ordered smallest ->
# largest. This is a capability comparison, not a comparison across model
# generations over time, and the thesis reports it as such.
MODEL_CATALOG = [
    {"label": "Claude Haiku 4.5", "provider": "anthropic", "model": "claude-haiku-4-5-20251001", "tier": 1},
    {"label": "Claude Sonnet 5",  "provider": "anthropic", "model": "claude-sonnet-5",           "tier": 2},
    {"label": "Claude Opus 5",    "provider": "anthropic", "model": "claude-opus-5",             "tier": 3},
]

# ----------------------------------------------------------------------------
# Item pool
# ----------------------------------------------------------------------------


PAPER_ITEMS = [
    "\"Consistency Regularisation for Unsupervised Domain Adaptation in Monocular Depth Estimation\" (2024): Monocular depth estimation models trained on one domain typically degrade sharply on a new, unlabelled target domain, and existing unsupervised domain adaptation solutions usually require training multiple separate models or complex multi-stage protocols. This work reframes adaptation as a consistency-based semi-supervised task: full supervision only on the labelled source domain, while target-domain predictions are encouraged to stay consistent across augmented views of the same input, removing the need for target-domain labels or auxiliary networks. Evaluated on KITTI and NYUv2 under several domain shifts, it matches or exceeds prior methods with a substantially simpler single-model pipeline.",
    "\"Uncertainty-Aware Likelihood Ratio Estimation for Pixel-Wise Out-of-Distribution Detection\" (2025): Segmentation models used in autonomous driving confidently mislabel unknown objects as known classes; pixel-wise OoD detection addresses this, but existing outlier-exposure methods struggle to separate truly unknown objects from merely rare known classes. This paper introduces an uncertainty-aware likelihood-ratio test using an evidential classifier that outputs full probability distributions, explicitly separating uncertainty from rare training examples versus imperfect synthetic outliers. On five urban-driving OoD benchmarks it achieves the lowest average false-positive rate among state-of-the-art methods while keeping high average precision, at negligible extra cost.",
    "\"GroupEnsemble: Efficient Uncertainty Estimation for DETR-based Object Detection\" (2026): Reliable object detectors need both semantic and spatial uncertainty, but standard ensemble methods (e.g. Deep Ensembles) require several independently trained models. GroupEnsemble instead feeds diverse groups of object queries through one shared DETR decoder in a single forward pass, with an attention mask keeping groups independent so each acts as an ensemble member -- yielding ensemble-style uncertainty without extra training cost. On Cityscapes and COCO, a hybrid with MC-Dropout outperforms full Deep Ensembles at a fraction of the computational cost.",
    "\"Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability\" (2026): Diffusion models can memorize and reproduce near-exact training images, raising privacy and copyright concerns. Existing norm-based memorization detectors are shown here to only work reliably when the log-probability landscape is roughly isotropic -- not true at the low noise levels most relevant to fine detail. Analysing this anisotropic regime, memorized samples show strong angular alignment between guidance and unconditional scores; combining this with the existing norm-based signal gives a detector computable from pure noise via two forward passes, ~5x faster than the previous best denoising-free method, and usable to guide a mitigation strategy.",
    "\"Forecasting the Past: Gradient-Based Distribution Shift Detection in Trajectory Prediction\" (2026): Trajectory-prediction models for automated driving can silently fail on out-of-distribution situations. This paper proposes a lightweight, post-hoc detector that never touches the original model: a decoder is trained after the fact on the self-supervised task of forecasting the second half of an observed trajectory from its first half, and the gradient norm of this loss becomes a distribution-shift score. Validated on the Shifts and Argoverse benchmarks, it substantially improves shift detection, and can also give early warning of simulated collisions for a motion planner in a highway simulator.",
    "\"Diffusion Model Guided Sampling with Pixel-Wise Aleatoric Uncertainty Estimation\" (2025): Diffusion models lack a cheap, built-in signal for per-pixel uncertainty during sampling, and existing uncertainty methods (MC-Dropout, ensembles) are impractical at scale. This paper proposes a training-free method estimating pixel-wise aleatoric uncertainty from the sensitivity of denoising output to small input perturbations, theoretically linked to the second derivative of the noise distribution. Rather than only reporting uncertainty, the method steers sampling toward high-uncertainty regions, improving final image quality (FID) over standard sampling on ImageNet and CIFAR-10.",
]

# The scenario used by every experiment. Which paper an assistant surfaces
# affects which research gets seen, so the task is deliberately the one that
# carries that consequence: choosing the primary reference for a claim.
SCENARIO = ("A researcher is writing the related-work section of a paper on "
            "uncertainty and robustness estimation in deep learning and needs to "
            "choose ONE paper to cite as the primary reference for this specific claim.")


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"title", "abstract"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing required column(s) {missing}. "
                             f"Found: {reader.fieldnames}")
        rows = list(reader)
    if not 2 <= len(rows) <= MAX_ITEMS:
        raise SystemExit(f"{path}: found {len(rows)} papers; the protocol shows "
                         f"the whole pool in one prompt, so it must hold "
                         f"between 2 and {MAX_ITEMS} papers.")
    return rows


def get_items() -> list[str]:
    """The candidate pool. Reads EXP_ITEMS_FILE if set, else PAPER_ITEMS.

    Every trial shows the WHOLE pool; only its order changes. Abstract
    lengths are checked because uneven length is exactly the confound
    Experiment 5 manipulates on purpose.
    """
    path = os.environ.get("EXP_ITEMS_FILE", "").strip()
    if not path:
        return PAPER_ITEMS
    rows = _read_rows(path)
    lengths = [len(r["abstract"].split()) for r in rows]
    if max(lengths) > 1.5 * min(lengths):
        print(f"  [warning] abstract lengths vary a lot ({min(lengths)}-"
              f"{max(lengths)} words); uneven length is the manipulation "
              f"tested in Experiment 5 and would confound the others.")
    items = []
    for r in rows:
        year = f" ({r['year']})" if r.get("year") else ""
        items.append(f"\"{r['title'].strip()}\"{year}: {r['abstract'].strip()}")
    print(f"Loaded {len(items)} papers from {path}")
    return items


def get_popularity_cues() -> list[str] | None:
    """One popularity cue per item, built from the REAL citation counts in
    the 'citations' column of EXP_ITEMS_FILE. Used by Experiment 4; returns
    None when no such column exists, so the experiment falls back to its
    fixed cue.

    The wording is deliberately neutral ("Cited N times to date.") rather
    than evaluative: real counts can be low for recent papers, and a claim
    like "highly cited" would be false in that case while also smuggling in
    a second manipulation on top of the number itself.
    """
    path = os.environ.get("EXP_ITEMS_FILE", "").strip()
    if not path:
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "citations" not in (reader.fieldnames or []):
            return None
        rows = list(reader)
    cues = []
    for r in rows:
        c = r.get("citations", "").strip()
        if not c:
            cues.append("")
        else:
            n = int(c)
            cues.append(f" Cited {n} {'time' if n == 1 else 'times'} to date.")
    return cues


# ----------------------------------------------------------------------------
# LLM client -- one function, three providers, plain HTTP
# ----------------------------------------------------------------------------

_last_call_ts: dict[str, float] = {}
_no_temperature: set[str] = set()   # models that reject "temperature"
_no_thinking: set[str] = set()      # models that reject "thinking"


def _pace(provider: str) -> None:
    """Sleep just enough to respect this provider's rate limit. Tracked per
    provider so Experiment 6, which queries several models in one run,
    doesn't force one provider's pacing onto another."""
    interval = MIN_INTERVAL if provider == PROVIDER else _DEFAULT_INTERVAL.get(provider, 2.0)
    wait = interval - (time.time() - _last_call_ts.get(provider, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_call_ts[provider] = time.time()


def _endpoint_and_headers(provider: str) -> tuple[str, dict]:
    def key(env: str) -> str:
        k = os.environ.get(env, "")
        if not k:
            raise SystemExit(f"{env} is not set for provider '{provider}'.")
        return k

    if provider == "together":
        return ("https://api.together.xyz/v1/chat/completions",
                {"Authorization": f"Bearer {key('TOGETHER_API_KEY')}"})
    if provider == "groq":
        return ("https://api.groq.com/openai/v1/chat/completions",
                {"Authorization": f"Bearer {key('GROQ_API_KEY')}"})
    if provider == "anthropic":
        return ("https://api.anthropic.com/v1/messages",
                {"x-api-key": key("ANTHROPIC_API_KEY"),
                 "anthropic-version": "2023-06-01"})
    raise SystemExit(f"Unknown provider '{provider}'. Use together, groq or anthropic.")


def call_llm(user_text: str, system: str, max_tokens: int = 64,
             retries: int = 5, provider: str | None = None,
             model: str | None = None) -> str:
    """One chat completion, returned as raw text. Retries on 429/5xx.

    Uses the globally configured provider/model unless `provider`/`model`
    are given explicitly, which is how Experiment 6 queries several models
    inside a single run.
    """
    eff_provider = provider or PROVIDER
    eff_model = model or (MODEL if eff_provider == PROVIDER
                          else _DEFAULT_MODELS.get(eff_provider, MODEL))
    url, headers = _endpoint_and_headers(eff_provider)
    headers = {**headers, "Content-Type": "application/json"}

    if eff_provider == "anthropic":
        # Claude models spend hidden reasoning tokens out of the same
        # max_tokens budget as the visible answer, so a 5-token budget for a
        # one-letter reply can come back empty. Give them headroom; the
        # parsers below find the letter anywhere in the response.
        payload = {
            "model": eff_model, "max_tokens": max(max_tokens, 150),
            "system": system,
            "messages": [{"role": "user", "content": user_text}],
        }
        if eff_model not in _no_temperature:
            payload["temperature"] = TEMPERATURE
        if eff_model not in _no_thinking:
            # Adaptive thinking is on by default when this field is omitted.
            # Picking a lettered option needs no reasoning, and leaving it on
            # spends the visible-answer budget, so it is disabled explicitly.
            payload["thinking"] = {"type": "disabled"}
    else:
        payload = {
            "model": eff_model, "max_tokens": max(max_tokens, 24),
            "temperature": TEMPERATURE,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user_text}],
        }

    for attempt in range(retries):
        _pace(eff_provider)
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"  [warn] network error ({e.__class__.__name__}), retrying...")
            continue

        if r.status_code == 429 or r.status_code >= 500:
            if attempt == retries - 1:
                raise SystemExit(f"API error {r.status_code}: {r.text[:300]}")
            time.sleep(max(10, 2 ** (attempt + 2)))
            continue

        if r.status_code != 200:
            # Some Claude releases reject "temperature", and some reject
            # "thinking". Drop the offending field once and keep going
            # rather than aborting a run over one parameter.
            low = r.text.lower()
            if r.status_code == 400 and eff_provider == "anthropic":
                if "temperature" in low and "temperature" in payload:
                    _no_temperature.add(eff_model)
                    del payload["temperature"]
                    continue
                if "thinking" in low and "thinking" in payload:
                    _no_thinking.add(eff_model)
                    del payload["thinking"]
                    continue
            raise SystemExit(f"API error {r.status_code}: {r.text[:300]}")

        data = r.json()
        if eff_provider == "anthropic":
            return "".join(b.get("text", "") for b in data.get("content", [])
                           if b.get("type") == "text").strip()
        content = data["choices"][0]["message"].get("content")
        if content is None:
            # Budget spent before anything visible was emitted. Return "" so
            # the caller treats the trial as unparseable and logs it, instead
            # of losing the whole run.
            print("  [warn] empty completion; trial treated as unparseable.")
            return ""
        return content.strip()
    return ""


# ----------------------------------------------------------------------------
# Prompt building and parsing
# ----------------------------------------------------------------------------

SYSTEM_SINGLE = (
    "You are a neutral recommendation assistant. Answer with exactly one "
    "capital letter (the option label) and nothing else -- no explanation, "
    "no lead-in words, just the single letter."
)

SYSTEM_RANKING = (
    "You are a neutral recommendation assistant. Answer with a ranking of ALL "
    "option letters from most to least recommended, as a comma-separated list "
    "(e.g. 'C,A,D,B,F,E') and nothing else -- no explanation, no lead-in words."
)


def shuffled_with_map(items: list[str], rng: random.Random) -> list[tuple[int, str]]:
    """Return [(item_id, text), ...] in random order. The item_id is the
    item's index in the pool, so a decision stays traceable to the paper
    that was chosen and not only to the slot it sat in."""
    indexed = list(enumerate(items))
    rng.shuffle(indexed)
    return indexed


def build_single_pick_prompt(ordered: list[tuple[int, str]], scenario: str = SCENARIO) -> str:
    lines = [scenario, "", "Candidate options:", ""]
    for slot, (_, text) in enumerate(ordered):
        lines.append(f"({LETTERS[slot]}) {text}")
    lines += ["", "Which option do you recommend? Reply with only the single letter."]
    return "\n".join(lines)


def build_ranking_prompt(ordered: list[tuple[int, str]], scenario: str = SCENARIO) -> str:
    lines = [scenario, "", "Candidate options:", ""]
    for slot, (_, text) in enumerate(ordered):
        lines.append(f"({LETTERS[slot]}) {text}")
    lines += ["", "Rank ALL options from most to least recommended. "
                  "Reply with only the comma-separated letters."]
    return "\n".join(lines)


def parse_single_letter(raw: str, n_items: int) -> int | None:
    """Return the chosen slot index (0-based), or None if unparseable.

    Patterns are tried from most to least specific so that a chatty answer
    ("I recommend option (C) because...") is not misparsed by matching a
    stray A-H character inside an ordinary word.
    """
    text = raw.upper()
    m = re.search(r"\(([A-H])\)", text) or re.search(r"\b([A-H])[).:]", text)
    if m:
        slot = LETTERS.index(m.group(1))
        return slot if slot < n_items else None
    # Last resort: a standalone letter. Take the LAST one -- explanatory
    # text builds up to the answer at the end, so an earlier standalone
    # capital is more likely to be the article "A".
    matches = re.findall(r"\b([A-H])\b", text)
    if not matches:
        return None
    slot = LETTERS.index(matches[-1])
    return slot if slot < n_items else None


def parse_ranking(raw: str, n_items: int) -> list[int] | None:
    """Return slot indices in ranked order, or None if the ranking is not
    a complete permutation. An exact run of n_items comma-separated letters
    (the format the prompt asks for) is preferred over scanning the whole
    text, for the same reason as parse_single_letter."""
    text = raw.upper()
    run = re.search(r"\b([A-H](?:\s*,\s*[A-H]){" + str(n_items - 1) + r"})\b", text)
    letters = re.findall(r"[A-H]", run.group(1)) if run else re.findall(r"[A-H]", text)
    slots: list[int] = []
    for ch in letters:
        s = LETTERS.index(ch)
        if s < n_items and s not in slots:
            slots.append(s)
    return slots if len(slots) == n_items else None


# ----------------------------------------------------------------------------
# Result persistence
# ----------------------------------------------------------------------------

def save_results(name: str, payload: dict,
                 models_under_test: list[dict] | None = None) -> Path:
    """Write the full raw trial log plus computed metrics and run metadata.
    Model and temperature are recorded with every file so that any reported
    number can be traced back to the exact configuration that produced it.

    `models_under_test` is for experiments that call several models
    explicitly rather than the session default (Experiment 6). Without it,
    "model" below would name the session default and misdescribe the run.
    """
    payload = dict(payload)
    payload["_meta"] = {
        "provider": PROVIDER,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if models_under_test is not None:
        payload["_meta"]["models_under_test"] = models_under_test
        payload["_meta"]["note"] = (
            "this experiment queries each model in models_under_test "
            "explicitly; 'provider' and 'model' above are the session "
            "defaults and are NOT the models these results come from")
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved raw results -> {out}")
    return out
