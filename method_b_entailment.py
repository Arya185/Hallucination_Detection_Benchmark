import json
import os
import re
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


# ============================================================
# LLM call
# ============================================================

def call_llm(prompt: str, temperature: float = 0.0) -> str:

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
# Method B
# ============================================================

def run_method_b():

    print(
        f"Evaluating {len(dataset)} claims "
        f"using Method B (NLI-style Entailment)..."
    )

    results_b = []

    for item in tqdm(dataset):

        source_text = item["source_text"]
        claim_text = item["claim_text"]

        prompt = f"""You are a strict fact-checking assistant.

Determine whether the provided claim is directly and factually
supported by the source text.

Source:
{source_text}

Claim:
"{claim_text}"

Task:
1. Is this claim directly and unambiguously supported by the source text?
2. Do not infer facts that are not stated or clearly entailed.
3. Treat exaggerated, causal, or stronger statements as unsupported
   unless the source explicitly supports them.

Format your response exactly as:

VERDICT: [YES or NO]
JUSTIFICATION: [Brief 1-2 sentence explanation of why it is or isn't entailed]
"""

        raw_response = call_llm(
            prompt,
            temperature=0.0
        )

        # ----------------------------------------------------
        # Parse VERDICT
        # ----------------------------------------------------

        verdict_match = re.search(
            r"VERDICT:\s*(YES|NO)",
            raw_response,
            re.IGNORECASE
        )

        if verdict_match:

            verdict_str = verdict_match.group(1).upper()

            is_supported = (verdict_str == "YES")

        else:

            # Fallback if response format varies
            first_part = raw_response[:100].upper()

            if "NO" in first_part:
                is_supported = False
            elif "YES" in first_part:
                is_supported = True
            else:
                # Conservative fallback:
                # if the model does not provide a valid verdict,
                # treat it as unsupported.
                is_supported = False

        # ----------------------------------------------------
        # Method B prediction
        # ----------------------------------------------------
        # NO = unsupported = caught

        method_b_caught = not is_supported

        # ----------------------------------------------------
        # Extract justification
        # ----------------------------------------------------

        justification_match = re.search(
            r"JUSTIFICATION:\s*(.*)",
            raw_response,
            re.DOTALL | re.IGNORECASE
        )

        if justification_match:

            justification = (
                justification_match.group(1).strip()
            )

        else:

            justification = raw_response.strip()

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results_b.append({

            "claim_id": item["claim_id"],

            "doc_id": item["doc_id"],

            "category": item["category"],

            "claim_text": claim_text,

            "is_unsupported_ground_truth":
                item["is_unsupported"],

            "raw_response": raw_response,

            "justification": justification,

            "method_b_caught":
                method_b_caught
        })

    # ========================================================
    # Save results
    # ========================================================

    output_path = "method_b_results.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results_b,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved Method B results to {output_path}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_method_b()