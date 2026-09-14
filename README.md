# Self-Consistency vs. Source-Grounded NLI for Hallucination Detection

Empirical comparison of two black-box hallucination-detection strategies for LLM-generated summaries — measuring where behavioral self-consistency checking succeeds and fails relative to direct source-grounded entailment.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![OpenAI SDK](https://img.shields.io/badge/OpenAI%20SDK-via%20OpenRouter-black)
![tqdm](https://img.shields.io/badge/tqdm-4.70.1-yellow)
![python-dotenv](https://img.shields.io/badge/python--dotenv-1.2.3-green)
![Model](https://img.shields.io/badge/model-gpt--4o--mini-informational)

---

## 1. Key Findings

**Research question:** for claims injected into LLM-generated summaries, how does the catch rate of black-box self-consistency checking (**Method A**) vary across claim type, and does source-grounded entailment (**Method B**) catch a distinct set of failures?

**Model under test:** `openai/gpt-4o-mini`, called through the OpenRouter API for both generation and verification.

### Catch rate (recall) on injected unsupported claims

| Category | N | Method A (Self-Consistency) | Method B (Source-Grounded NLI) |
|---|:---:|:---:|:---:|
| Numerical fabrication | 11 | 100.0% (11/11) | 100.0% (11/11) |
| Entity fabrication | 10 | 90.0% (9/10) | 100.0% (10/10) |
| Causal overreach | 25 | 96.0% (24/25) | 100.0% (25/25) |
| **Overall** | **46** | **95.7% (44/46)** | **100.0% (46/46)** |

### False positive rate on the 86 supported controls

| Method | False Positives | FPR |
|---|:---:|:---:|
| Method A (Self-Consistency) | 5 / 86 | 5.8% |
| Method B (Source-Grounded NLI) | 4 / 86 | 4.7% |

### Cross-tabulation (unsupported claims only)

| | Method B Caught | Method B Missed |
|---|:---:|:---:|
| **Method A Caught** | 44 | 0 |
| **Method A Missed** | 2 | 0 |

**Takeaway:** Method B strictly dominates on recall (it never misses what Method A misses, and misses nothing itself), because it has direct access to the reference text at inference time. Method A, operating purely on black-box output consistency with no access to the source document, still recovers ~96% of injected claims — but leaks specifically on entity attribution and causal-hyperbole cases where an equivalence checker mistakes shared rhetorical framing for factual corroboration.

---

## 2. Repository Architecture

```
.
├── claims_dataset.py           # Defines the 132-claim adversarial benchmark (13 source docs)
├── method_a_self_consistency.py# Method A: N=5 regeneration + equivalence-agreement scoring
├── method_b_entailment.py      # Method B: single-pass zero-shot NLI against source text
├── evaluate_and_score.py       # Scores both methods against ground truth, builds confusion matrix
├── check_missed_claims.py      # Prints qualitative detail on claims Method A missed
├── research_report.md          # Narrative write-up of methodology, results, and error analysis
│
├── claims_dataset.json         # Dataset dump (written only if claims_dataset.py is run directly)
├── regenerations_cache.json    # Cached N=5 summaries per document (Method A intermediate output)
├── method_a_results.json       # Per-claim Method A verdicts + agreement scores
├── method_b_results.json       # Per-claim Method B verdicts + parsed justifications
└── evaluation_summary.json     # Aggregated catch-rate / FPR / cross-tabulation, written by evaluate_and_score.py
```

Script and filename mapping above reflects the code exactly as committed; no wrapper scripts, CLI flags, or additional entry points exist beyond what is listed.

---

## 3. Methodology

### 3.1 Adversarial dataset (`claims_dataset.py`)

The benchmark is a hand-authored Python list (`dataset`, 132 entries) spanning **13 source documents** drawn from astronomy press releases, corporate earnings reports, public-policy/municipal orders, biomedical trial summaries, and hardware announcements. Each entry carries `claim_id`, `doc_id`, `source_text`, `claim_text`, `category`, and `is_unsupported`.

Claims fall into five category labels, split into two evaluation groups:

- **Unsupported injections (N=46)** — claims fabricated relative to the source:
  - `numerical` (N=11): altered metrics, dates, percentages.
  - `entity` (N=10): fabricated researchers, agencies, institutions.
  - `causal` (N=25): unsupported causal links or guaranteed outcomes.
- **Supported controls (N=86)** — claims grounded in or plausibly hedged against the source, spanning the `numerical`, `entity`, `causal`, `hedged_plausible`, and `supported_control` labels wherever `is_unsupported` is `False`.

Running `claims_dataset.py` directly (`if __name__ == "__main__"`) serializes the list to `claims_dataset.json` via `save_dataset()`. All other scripts import `dataset` directly from the module (`from claims_dataset import dataset`), so this dump is for inspection only and is not required for the pipeline to run.

### 3.2 Method A — Self-Consistency (`method_a_self_consistency.py`)

Purely black-box: never sees the source document at verification time.

1. For each of the 13 source documents, generate `N_REGENERATIONS = 5` alternative summaries at `temperature=0.7` (cached to `regenerations_cache.json` so reruns skip regeneration).
2. For each of the 132 claims, run a `YES`/`NO` equivalence prompt against each of the 5 regenerations at `temperature=0.0`, asking whether that regeneration mentions or supports the claim.
3. Compute `agreement_score = match_count / 5`.
4. Flag the claim as caught (predicted unsupported) if `agreement_score <= AGREEMENT_THRESHOLD` (`0.40`, i.e. recurrence in ≤2/5 generations).

### 3.3 Method B — Source-Grounded NLI (`method_b_entailment.py`)

Single-pass, given the actual source text. For each claim, a zero-shot entailment prompt at `temperature=0.0` asks whether the source text directly and unambiguously supports the claim, instructing the model to treat exaggerated or causal overreach as unsupported absent explicit textual backing. The response is parsed with regex for a `VERDICT: YES|NO` line and a `JUSTIFICATION:` line (with a conservative fallback to "unsupported" if no clean verdict is parsed). `VERDICT: NO` → `method_b_caught = True`.

### 3.4 Scoring (`evaluate_and_score.py`, `check_missed_claims.py`)

`evaluate_and_score.py` joins `method_a_results.json` and `method_b_results.json` on `claim_id`, computes per-category recall, the overall false-positive rate on the 86 controls, and the 2×2 cross-tabulation of catch/miss between methods on the 46 unsupported claims, writing the aggregate to `evaluation_summary.json`. `check_missed_claims.py` reads that summary's `b_only` list (claims Method B caught and Method A missed) and prints each claim's text and agreement score for qualitative review.

---

## 4. Detailed Results & Failure Autopsy

Method A missed exactly 2 of the 46 injected claims (both caught by Method B), yielding its 95.7% overall recall:

**`doc11_c04` — Entity fabrication, agreement score 0.8 (4/5)**
Claim attributes a specific legal opinion to a fabricated municipal attorney. All 5 regenerations converged on discussing the underlying municipal legal topic, and the equivalence checker treated topical overlap as support for the fabricated attribution — a case of pre-training priors smoothing over generic local-government role conventions rather than verifying the specific named actor.

**`doc13_c06` — Causal overreach, agreement score 1.0 (5/5)**
Claim asserts a discovery "would have been completely impossible with any other existing telescope." Every regeneration praised the relevant instrument's role in the discovery, and the checker interpreted that shared rhetorical enthusiasm as corroborating the absolute exclusivity claim — self-consistency cannot distinguish genuine factual entailment from stylistically-correlated hyperbole that recurs across independent generations for the same underlying reason.

Both failures share a mechanism: self-consistency measures whether independent generations *agree*, not whether they are *individually grounded* in the source. When a fabrication rides on a topic or rhetorical framing the base model reliably reproduces regardless of source grounding, agreement stays high and the claim slips through. Source-grounded NLI (Method B) is immune to this failure mode by construction, at the cost of requiring the reference text at inference time — Method A requires no such access, at ~5x+ the inference cost (5 summary generations + 5 equivalence checks per claim, vs. one entailment call per claim for Method B).

---

## 5. Quickstart & Reproducibility

`method_a_self_consistency.py` and `method_b_entailment.py` load their OpenRouter credential via `python-dotenv` (`load_dotenv()` + `os.getenv("OPENROUTER_API_KEY")`), reading it from a local `.env` file that is excluded from version control by `.gitignore`. No key is stored in source.

```bash
# 1. Activate the existing virtual environment
source .venv/bin/activate

# 2. Install the one additional dependency this env-var change introduced
pip install python-dotenv

# 3. Provide your own credential — copy the template and fill in your key
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=<your-openrouter-key>

# 4. Run the pipeline in order — each step reads/writes the JSON artifacts
#    documented in the Repository Architecture section above.

# (optional) materialize the dataset to disk for inspection
python claims_dataset.py

# Method A: generates regenerations_cache.json, then method_a_results.json
python method_a_self_consistency.py

# Method B: generates method_b_results.json
python method_b_entailment.py

# Score both methods against ground truth -> evaluation_summary.json
python evaluate_and_score.py

# Print qualitative detail on the 2 claims Method A missed
python check_missed_claims.py
```

`method_a_self_consistency.py` and `evaluate_and_score.py` check for their respective cached JSON files (`regenerations_cache.json`, `method_a_results.json`, `method_b_results.json`) before making any API calls — delete the relevant file to force a fresh run of that stage, or leave the existing cached files in place to re-score or re-inspect results with zero new API calls.

**Security note:** this repository previously had a live OpenRouter key hardcoded in `method_a_self_consistency.py` and `method_b_entailment.py`. That has been remediated — both scripts now read `OPENROUTER_API_KEY` from `.env` (gitignored; `.env.example` documents the expected variable with a placeholder value). The key that was previously hardcoded in source has already been exposed and should still be treated as compromised and rotated in the OpenRouter dashboard, independent of this code fix.

---

## 6. Threats to Validity

- **Single-model dependency.** Both generation (Method A's regenerations) and both verification steps (Method A's equivalence checks, Method B's entailment checks) run on the same model, `gpt-4o-mini`. Results may not transfer to verification with a stronger or differently-trained judge model, or to detecting fabrications produced by a different generator.
- **Dataset scale and construction.** 132 claims across 13 documents were hand-authored by a single author with no multi-annotator agreement measurement (e.g., Cohen's Kappa) on the unsupported/supported labels themselves.
- **Threshold sensitivity.** Method A's catch decision depends on a fixed `AGREEMENT_THRESHOLD = 0.40` (≤2/5 regenerations) and `N_REGENERATIONS = 5`, chosen a priori rather than tuned. The reported 95.7% recall / 5.8% FPR trade-off is a single point on what is otherwise a tunable precision–recall curve; different N or threshold values would shift both catch rate and false-positive rate.
# Hallucination_Detection_Benchmark
