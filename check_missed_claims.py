import json

with open("evaluation_summary.json") as f:
    eval_data = json.load(f)

with open("method_a_results.json") as f:
    results_a = {x["claim_id"]: x for x in json.load(f)}

print("=== METHOD A MISSED CLAIMS (Caught by B only) ===")
for cid in eval_data["cross_tabulation"]["items"]["b_only"]:
    item = results_a[cid]
    print(f"\nClaim ID: {cid} | Category: {item['category']}")
    print(f"Claim: {item['claim_text']}")
    print(f"Agreement Score: {item['agreement_score']} (Matches: {item['matches']})")