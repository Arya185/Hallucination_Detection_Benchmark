import json
import os
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from claims_dataset import dataset

# ============================================================
# OpenRouter / OpenAI-compatible client
# ============================================================

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "openai/gpt-4o-mini"

N_REGENERATIONS = 5

AGREEMENT_THRESHOLD = 0.40
# Flag as unsupported if agreement <= 0.40


# ============================================================
# LLM call
# ============================================================

def call_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    Call an LLM through OpenRouter using the OpenAI-compatible API.
    """

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
    )

    return resp.choices[0].message.content.strip()


# ============================================================
# Method A
# ============================================================

def run_method_a():

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    # dataset is imported from claims_dataset.py

    docs = {}

    for item in dataset:
        docs[item["doc_id"]] = item["source_text"]

    # --------------------------------------------------------
    # 2. Generate N alternative summaries per document
    # --------------------------------------------------------

    regenerations_cache_path = "regenerations_cache.json"

    if os.path.exists(regenerations_cache_path):

        with open(
            regenerations_cache_path,
            "r",
            encoding="utf-8"
        ) as f:
            regenerations = json.load(f)

        print(
            f"Loaded cached regenerations for "
            f"{len(regenerations)} docs."
        )

    else:

        print(
            f"Generating {N_REGENERATIONS} summaries per document..."
        )

        regenerations = defaultdict(list)

        for doc_id, text in tqdm(docs.items()):

            prompt = (
                "Summarize the key facts and statements in the "
                "following document concisely in 3-4 paragraphs:\n\n"
                f"{text}"
            )

            for _ in range(N_REGENERATIONS):

                summary = call_llm(
                    prompt,
                    temperature=0.7
                )

                regenerations[doc_id].append(summary)

        with open(
            regenerations_cache_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                dict(regenerations),
                f,
                indent=2,
                ensure_ascii=False
            )

    # --------------------------------------------------------
    # 3. Check claim presence across regenerations
    # --------------------------------------------------------

    print(
        "Evaluating claim recurrence across regenerations..."
    )

    results_a = []

    for item in tqdm(dataset):

        doc_id = item["doc_id"]
        claim_text = item["claim_text"]

        summaries = regenerations[doc_id]

        match_count = 0
        per_run_matches = []

        for summary in summaries:

            eval_prompt = (
                f"Summary:\n{summary}\n\n"
                f'Claim: "{claim_text}"\n\n'
                "Does the summary above mention or directly support "
                "this claim?\n"
                "Answer with only 'YES' or 'NO'."
            )

            response = call_llm(
                eval_prompt,
                temperature=0.0
            ).strip().upper()

            is_present = response == "YES"

            per_run_matches.append(is_present)

            if is_present:
                match_count += 1

        agreement_score = (
            match_count / N_REGENERATIONS
        )

        # Flag as unsupported if it does not consistently recur
        predicted_unsupported = (
            agreement_score <= AGREEMENT_THRESHOLD
        )

        results_a.append({
            "claim_id": item["claim_id"],
            "doc_id": doc_id,
            "category": item["category"],
            "claim_text": claim_text,
            "is_unsupported_ground_truth":
                item["is_unsupported"],
            "agreement_score": agreement_score,
            "matches": per_run_matches,
            "method_a_caught": predicted_unsupported
        })

    # --------------------------------------------------------
    # 4. Save results
    # --------------------------------------------------------

    results_path = "method_a_results.json"

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results_a,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved Method A results to {results_path}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_method_a()