import json
from collections import defaultdict

def load_results():
    with open("method_a_results.json", "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open("method_b_results.json", "r", encoding="utf-8") as f:
        data_b = json.load(f)
    return data_a, data_b

def evaluate():
    data_a, data_b = load_results()
    
    # Map by claim_id
    map_b = {item["claim_id"]: item for item in data_b}
    
    # Metrics containers
    # categories: numerical, entity, causal, hedged_plausible, supported_control
    unsupported_counts = defaultdict(int)
    caught_a_counts = defaultdict(int)
    caught_b_counts = defaultdict(int)
    
    control_total = 0
    control_fp_a = 0
    control_fp_b = 0
    
    # Confusion matrix counters (on unsupported claims only)
    both_caught = 0
    a_only_caught = 0
    b_only_caught = 0
    neither_caught = 0
    
    cross_tab_items = {
        "both": [],
        "a_only": [],
        "b_only": [],
        "neither": []
    }

    for item_a in data_a:
        cid = item_a["claim_id"]
        item_b = map_b[cid]
        
        cat = item_a["category"]
        is_unsupported = item_a["is_unsupported_ground_truth"]
        
        caught_a = item_a["method_a_caught"]
        caught_b = item_b["method_b_caught"]
        
        if is_unsupported:
            unsupported_counts[cat] += 1
            if caught_a:
                caught_a_counts[cat] += 1
            if caught_b:
                caught_b_counts[cat] += 1
                
            # Cross-tabulation
            if caught_a and caught_b:
                both_caught += 1
                cross_tab_items["both"].append(cid)
            elif caught_a and not caught_b:
                a_only_caught += 1
                cross_tab_items["a_only"].append(cid)
            elif not caught_a and caught_b:
                b_only_caught += 1
                cross_tab_items["b_only"].append(cid)
            else:
                neither_caught += 1
                cross_tab_items["neither"].append(cid)
        else:
            # Negative controls / supported claims
            control_total += 1
            if caught_a:
                control_fp_a += 1
            if caught_b:
                control_fp_b += 1

    # --- Print Results ---
    print("\n=======================================================")
    print("           CATCH RATE BY CATEGORY (RECALL)             ")
    print("=======================================================")
    print(f"{'Category':<20} | {'Total':<6} | {'Method A (SC)':<15} | {'Method B (NLI)':<15}")
    print("-" * 64)
    
    total_unsupported = sum(unsupported_counts.values())
    total_caught_a = sum(caught_a_counts.values())
    total_caught_b = sum(caught_b_counts.values())
    
    for cat in sorted(unsupported_counts.keys()):
        tot = unsupported_counts[cat]
        a_rec = (caught_a_counts[cat] / tot) * 100 if tot > 0 else 0
        b_rec = (caught_b_counts[cat] / tot) * 100 if tot > 0 else 0
        print(f"{cat:<20} | {tot:<6} | {a_rec:>6.1f}% ({caught_a_counts[cat]}/{tot}) | {b_rec:>6.1f}% ({caught_b_counts[cat]}/{tot})")
    
    print("-" * 64)
    overall_a = (total_caught_a / total_unsupported) * 100 if total_unsupported > 0 else 0
    overall_b = (total_caught_b / total_unsupported) * 100 if total_unsupported > 0 else 0
    print(f"{'OVERALL UNSUPPORTED':<20} | {total_unsupported:<6} | {overall_a:>6.1f}% ({total_caught_a}/{total_unsupported}) | {overall_b:>6.1f}% ({total_caught_b}/{total_unsupported})")
    
    print("\n=======================================================")
    print("         FALSE POSITIVE RATE (ON TRUE CONTROLS)        ")
    print("=======================================================")
    fpr_a = (control_fp_a / control_total) * 100 if control_total > 0 else 0
    fpr_b = (control_fp_b / control_total) * 100 if control_total > 0 else 0
    print(f"Total Controls: {control_total}")
    print(f"Method A FPR: {fpr_a:.1f}% ({control_fp_a}/{control_total})")
    print(f"Method B FPR: {fpr_b:.1f}% ({control_fp_b}/{control_total})")

    print("\n=======================================================")
    print("   CROSS-TABULATION MATRIX (UNSUPPORTED CLAIMS ONLY)   ")
    print("=======================================================")
    print(f"{'':<20} | {'Method B Caught':<18} | {'Method B Missed':<18}")
    print("-" * 62)
    print(f"{'Method A Caught':<20} | {both_caught:<18} | {a_only_caught:<18}")
    print(f"{'Method A Missed':<20} | {b_only_caught:<18} | {neither_caught:<18}")
    print("=======================================================\n")

    # Save summary artifact for the write-up
    summary_data = {
        "catch_rate_by_category": {
            cat: {
                "total": unsupported_counts[cat],
                "method_a": caught_a_counts[cat],
                "method_b": caught_b_counts[cat]
            } for cat in unsupported_counts
        },
        "controls": {
            "total": control_total,
            "false_positives_a": control_fp_a,
            "false_positives_b": control_fp_b
        },
        "cross_tabulation": {
            "both": both_caught,
            "a_only": a_only_caught,
            "b_only": b_only_caught,
            "neither": neither_caught,
            "items": cross_tab_items
        }
    }
    
    with open("evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("Saved evaluation summary to evaluation_summary.json")

if __name__ == "__main__":
    evaluate()