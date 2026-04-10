import json
import numpy as np
from metrics import (
    extract_json_from_text,
    extract_rationale_from_text,
    normalize_decision,
    get_optimal_matches,
    VALID_DECISIONS,
    VALID_TASK_TYPES
)

PREDICTIONS_FILE = "../data/results/d2v2x_results.json"
TOP_K = 5
USE_BERT = False

if USE_BERT:
    from bert_score import score as calc_bert_score

def find_best_cvpr_examples(preds, labels):
    print(f"Evaluating {len(preds)} samples to find the best figure examples...")
    
    # Extract rationales and optionally calculate BERTScore
    pred_rats = [extract_rationale_from_text(p) for p in preds]
    gt_rats = [extract_rationale_from_text(l) for l in labels]
    
    bert_scores = [0.0] * len(preds)
    if USE_BERT:
        print("Calculating BERTScores in batch...")
        _, _, B_F1_Tensor = calc_bert_score(
            pred_rats, gt_rats, lang="en", verbose=False, model_type="distilbert-base-uncased"
        )
        bert_scores = B_F1_Tensor.numpy()

    # Dictionary to group scored samples by task type
    task_buckets = {task: [] for task in VALID_TASK_TYPES}
    task_buckets["unknown"] = []

    # Iterate through and calculate sample-level metrics
    print("Parsing structured JSON and calculating spatial metrics...")
    for idx, (pred_text, label_text) in enumerate(zip(preds, labels)):
        pred_json = extract_json_from_text(pred_text)
        label_json = extract_json_from_text(label_text)

        if not isinstance(pred_json, dict): pred_json = {}
        if not label_json: continue

        task_type = label_json.get("task_type", "unknown").lower()
        if task_type not in task_buckets:
            task_type = "unknown"

        gt_decision = normalize_decision(label_json.get("decision"))
        pred_decision = normalize_decision(pred_json.get("decision") if pred_json else None)
        is_decision_correct = (gt_decision == pred_decision and gt_decision in VALID_DECISIONS)
        
        gt_objects = label_json.get("grounded_objects", [])
        pred_objects = pred_json.get("grounded_objects", []) if pred_json else []
        
        scene_ious, scene_maes_vis, scene_maes_occ, occ_tp, occ_fp, occ_fn, vis_fp = get_optimal_matches(gt_objects, pred_objects)

        # Calculate True Positives, False Positives, and False Negatives
        tp = len(scene_ious) + occ_tp
        fp = vis_fp + occ_fp
        fn = max(0, len(gt_objects) - tp)
        
        # Calculate Object F1-Score
        obj_f1 = (2 * tp) / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        # Aggregate spatial errors
        avg_iou = np.mean(scene_ious) if scene_ious else 0.0
        all_maes = scene_maes_vis + scene_maes_occ
        avg_mae = np.mean(all_maes) if all_maes else float('inf')

        if len(gt_objects) == 0 and len(pred_objects) == 0:
            continue

        task_buckets[task_type].append({
            "index": idx,
            "is_decision_correct": is_decision_correct,
            "obj_f1": obj_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "avg_iou": avg_iou,
            "avg_mae": avg_mae,
            "bert_score": bert_scores[idx],
            "gt_rat": gt_rats[idx],
            "pred_rat": pred_rats[idx],
            "pred_json": pred_json,
            "gt_json": label_json
        })

    # Sort and Print Top K for each task type
    print("\n" + "="*70)
    print("TOP EXAMPLES BY TASK TYPE")
    print("="*70)

    for task in VALID_TASK_TYPES:
        samples = task_buckets[task]
        if not samples:
            print(f"\nNo valid examples found for task: {task.upper()}")
            continue

        samples.sort(key=lambda x: (
            x["is_decision_correct"], 
            x["obj_f1"], 
            x["avg_iou"], 
            -x["avg_mae"], 
            x["bert_score"]
        ), reverse=True)

        print(f"\n" + "*"*60)
        print(f"TASK: {task.upper()} (Showing Top {min(TOP_K, len(samples))})")
        print("*"*60)

        for rank in range(min(TOP_K, len(samples))):
            sample = samples[rank]
            print(f"\n--- RANK {rank + 1} (Dataset Index {sample['index']}) ---")
            print(f"Decision Correct: {sample['is_decision_correct']}")
            print(f"Object Accuracy (F1): {sample['obj_f1']:.4f} [TP: {sample['tp']} | FP: {sample['fp']} | FN: {sample['fn']}]")
            print(f"Avg IoU: {sample['avg_iou']:.4f} | Avg Dist Error (MAE): {sample['avg_mae']:.2f}m")
            if USE_BERT:
                print(f"BERTScore: {sample['bert_score']:.4f}")
            print("\nGROUND TRUTH RATIONALE:")
            print(sample['gt_rat'])
            print("\nPREDICTED RATIONALE:")
            print(sample['pred_rat'])
            print("\nGROUND TRUTH JSON:")
            print(json.dumps(sample['gt_json'], indent=2))
            print("\nPREDICTED JSON:")
            print(json.dumps(sample['pred_json'], indent=2))
            print("-" * 60)

if __name__ == "__main__":
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            data = json.load(f)
            preds = [item["prediction"] for item in data]
            labels = [item["ground_truth"] for item in data]
            
        find_best_cvpr_examples(preds, labels)
    except FileNotFoundError:
        print(f"Please update PREDICTIONS_FILE path. Currently looking for: {PREDICTIONS_FILE}")