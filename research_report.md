# Research Report: Does Self-Consistency Catch Unsupported Claims?

## Abstract
This study evaluates whether black-box self-consistency checking (Method A) can reliably flag unsupported assertions injected into LLM-generated texts across numerical, entity, and causal categories, and contrasts its performance against direct source-grounded Natural Language Inference (Method B). On an adversarial test set of 132 evaluated items (46 unsupported injections, 86 supported controls) derived across 10 diverse technical and public-sector documents, Method B achieved a 100.0% catch rate (46/46), while Method A achieved a 95.7% catch rate (44/46) with a 5.8% false-positive rate. Self-consistency proved highly effective at catching numerical fabrications (100%), but exhibited edge-case leakage in entity and causal overreach claims.

---

## 1. Methodology & Experimental Setup

### 1.1 Dataset Construction
The test suite consists of 10 primary source documents (covering astronomy, corporate earnings, biomedical clinical trials, municipal ordinances, and hardware releases) and 132 granular claims:
- **Numerical Fabrications ($N=11$):** Altered quantities, percentages, sample sizes, and dates.
- **Entity Fabrications ($N=10$):** Fictitious researchers, partner institutions, or mission instruments.
- **Causal Overreach ($N=25$):** Extrapolations that convert correlation, launch announcements, or phase-trial starts into definitive causal efficacy or guaranteed future outcomes.
- **Negative Controls ($N=86$):** Directly supported factual claims and hedged statements reflecting the source ground truth.

### 1.2 Model & Methods Under Test
* **Model:** `openai/gpt-4o-mini` (via OpenRouter API).
* **Method A (Self-Consistency Checking):** 
  - For each source document, $N=5$ alternative summaries were generated at `temperature=0.7`.
  - For each claim, an equivalence prompt evaluated whether the claim was stated or directly entailed by each regeneration (`temperature=0.0`).
  - An agreement score $\in [0, 1]$ was computed. Claims with agreement score $\le 0.40$ (recurrence in $\le 2/5$ generations) were flagged as unsupported.
* **Method B (Source-Grounded NLI Checking):**
  - Evaluated the claim directly against the raw source text using a zero-shot factual entailment prompt at `temperature=0.0`.
  - Parsed binary verdicts (`YES` / `NO`) along with chain-of-thought justifications.

---

## 2. Experimental Results

### 2.1 Catch Rate (Recall) by Category

| Category | Injected Claims ($N$) | Method A (Self-Consistency) | Method B (NLI) |
| :--- | :---: | :---: | :---: |
| **Numerical Fabrication** | 11 | **100.0%** (11/11) | **100.0%** (11/11) |
| **Entity Fabrication** | 10 | **90.0%** (9/10) | **100.0%** (10/10) |
| **Causal Overreach** | 25 | **96.0%** (24/25) | **100.0%** (25/25) |
| **Overall Unsupported** | **46** | **95.7%** (44/46) | **100.0%** (46/46) |

### 2.2 False Positive Rate (Negative Controls)

| Metric | Method A (Self-Consistency) | Method B (NLI) |
| :--- | :---: | :---: |
| Supported Controls Tested | 86 | 86 |
| False Positives (Incorrectly Flagged) | 5 | 4 |
| **False Positive Rate (FPR)** | **5.8%** | **4.7%** |

### 2.3 Cross-Tabulation Matrix (Unsupported Claims)

| | Method B Caught | Method B Missed |
| :--- | :---: | :---: |
| **Method A Caught** | 44 | 0 |
| **Method A Missed** | 2 | 0 |

---

## 3. Failure Mode & Error Analysis

1. **Why Method B Strictly Outperforms on Recall:**
   Method B has direct access to the ground-truth reference text. As long as the prompt explicitly enforces strict factual entailment, `gpt-4o-mini` consistently detects when an assertion lacks direct textual basis.
2. **Why Method A Missed 2 Injected Claims (Leakage Mechanism):**
   - **Prior Bias and Common Associations:** In entity fabrications, if an injected entity aligns closely with widespread pre-training priors (e.g., plausible astronomical conventions or standard corporate titles), multiple independent summary regenerations can spontaneously generate the same semantic hallucination or accept it as natural context.
   - **Linguistic Alignment in Causal Claims:** Causal overreach expressed through natural summary rhetoric (e.g., attributing quarterly revenue strength to the primary segment highlighted in the prompt) recurrently appears across stochastic generations because the model repeatedly selects the most salient explanatory narrative.
3. **Operational Trade-offs:**
   - Method A is computationally expensive ($5\times$ summary generation calls $+ 5\times$ equivalence verifications per claim), but functions without retaining the original source text at inference time.
   - Method B is single-pass and significantly cheaper, but requires low-latency retrieval access to the full source context.

### 3.1 Qualitative Autopsy of Missed Claims (Method A False Negatives)

Method A failed to catch exactly two injected claims, yielding an overall recall of 95.7%:

1. **Entity/Attribution Conflation (`doc11_c04` — Agreement Score: 0.8 / 80%)**
   - *Injected Claim:* "City Attorney Jennifer Martinez advised the council that state law allows local age restrictions on e-bike operators."
   - *Ground Truth:* Injected entity and role; the source only notes that councilmembers asked staff to explore legislative options.
   - *Mechanism:* Regenerations consistently highlighted municipal legal debates. The equivalence verifier exhibited semantic drift, accepting a fabricated municipal actor as supported context because the topical discourse matched.

2. **Hyperbolic Exclusivity (`doc13_c06` — Agreement Score: 1.0 / 100%)**
   - *Injected Claim:* "Webb's coronagraph technology enabled this discovery, which would have been completely impossible with any other existing telescope."
   - *Ground Truth:* Unsupported causal exclusivity; the source mentions coronagraph use without claiming unique impossibility across all other telescopes.
   - *Mechanism:* Every regeneration praised the coronagraph's instrumental role. The checker interpreted the consistent emphasis on the coronagraph as warranting the absolute causal claim, demonstrating that self-consistency cannot distinguish between genuine factual entailment and shared rhetorical hyperbole.
---

## 4. Threats to Validity & Limitations
- **Single Evaluator Model:** Both generation and verification were conducted using `gpt-4o-mini`. Cross-model verification (e.g., generating with Llama-3 and checking with Claude/GPT-4) was not measured.
- **Annotator Bias:** Claims were manually injected by a single researcher without multi-annotator inter-rater agreement (Cohen's Kappa).
- **Threshold Sensitivity:** The decision threshold for Method A was fixed at $\le 0.40$ *a priori*. While preventing post-hoc tuning, varying $N$ or the threshold could alter the FPR/recall curve.