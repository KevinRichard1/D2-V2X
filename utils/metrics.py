'''Module to calculate metrics'''
import re
import json
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
from bert_score import score as calc_bert_score

VALID_TASK_TYPES  = {"maneuver", "counting", "spatial"}
VALID_DECISIONS = ["yield", "monitor", "safe", "unsafe"]
DECISION_SYNONYMS = {
    # yield variants
    "stop": "yield", "wait": "yield", "halt": "yield", "brake": "yield",
    "do not proceed": "yield", "do_not_proceed": "yield", "not clear": "yield",
    "not_clear": "yield", "path_not_clear": "yield", "not_safe": "yield",
    "do not merge": "yield", "do_not_merge": "yield", "do not switch lanes": "yield",
    "do_not_switch_lane": "yield", "do not turn": "yield", "do_not_turn": "yield",
    "avoid": "yield",
    # monitor variants
    "proceed": "monitor", "continue": "monitor", "go": "monitor",
    "proceed_with_caution": "monitor", "caution": "monitor",
    "maintain": "monitor", "maintain_speed": "monitor",
    "maintain distance": "monitor", "check": "monitor",
    "observe": "monitor", "analyze": "monitor", "detect": "monitor",
    "pass": "monitor",
    # safe variants
    "clear": "safe", "no_hazard": "safe", "no hazard": "safe",
    "no_hidden_vehicles": "safe", "no hidden vehicles": "safe",
    "no_hidden_cars": "safe", "no hidden cars": "safe",
    "no_hidden_trucks": "safe", "no hidden truck": "safe",
    "no_hidden_pedestrians": "safe", "no hidden pedestrians": "safe",
    "no_occluded": "safe", "no_occlusion": "safe", "no_occlusions": "safe",
    "no_obstacle": "safe", "no_obstruction": "safe",
    "zero_hidden": "safe", "zero hidden vehicles": "safe",
    "no_block": "safe", "no_concealment": "safe",
    "yes": "safe",
    # unsafe variants
    "hidden_vehicles": "unsafe", "hidden_cars": "unsafe",
    "hidden_pedestrians": "unsafe", "hidden_pedestrian": "unsafe",
    "hidden_car": "unsafe", "hidden_buses": "unsafe",
    "occluded": "unsafe", "occlusion_detected": "unsafe",
    "occluded_vehicle": "unsafe", "obscured_vehicles": "unsafe",
    "hidden": "unsafe", "detected": "unsafe",
    "no": "unsafe", "alert": "unsafe", "unsafe": "unsafe",
    "not visible": "unsafe",
}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def extract_json_from_text(text):
    '''Extracts and parses the JSON block from the model's text output.'''
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    
def extract_rationale_from_text(text):
    '''Extracts rationale'''
    match = re.search(r'<think>\s*(.*?)\s*</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    parts = text.split("```json")
    if len(parts) > 0 and parts[0].strip():
        return parts[0].strip()
    return "No rationale provided."

def normalize_decision(raw):
    '''Remap a raw decision string'''
    if raw is None:
        return "invalid_decision"
    raw = str(raw).strip().lower()
    return DECISION_SYNONYMS.get(raw, raw)

def calculate_iou(box1, box2):
    '''Calculates Intersection over Union for two bounding boxes'''
    if not isinstance(box1, (list, tuple)) or len(box1) != 4:
        return 0.0
    if not isinstance(box2, (list, tuple)) or len(box2) != 4:
        return 0.0
    
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = float(box1_area + box2_area - intersection_area)
    if union_area <= 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou

def get_optimal_matches(gt_objects, pred_objects, penalty_dist=100.0, dist_threshold=10.0):
    '''Pair Predicted boxes to ground truth boxes'''
    if not pred_objects:
        pred_objects = []
    
    if isinstance(pred_objects, dict):
        if "objects" in pred_objects and isinstance(pred_objects["objects"], list):
            pred_objects = pred_objects["objects"]
        else:
            pred_objects = [pred_objects]

    if not isinstance(pred_objects, list):
        pred_objects = []

    pred_objects = [obj for obj in pred_objects if isinstance(obj, dict)]

    N = len(gt_objects)
    M = len(pred_objects)

    scene_ious = []
    scene_maes_vis = []
    scene_maes_occ = []
    occ_tp = 0
    occ_fp = 0
    occ_fn = 0
    vis_fp = 0

    # Hallucination
    if N == 0 and M > 0:
        for p in pred_objects:
            if len(p.get("bbox", [])) != 4:
                occ_fp += 1
            else:
                vis_fp += 1
                scene_ious.append(0.0)
        return scene_ious, scene_maes_vis, scene_maes_occ, occ_tp, occ_fp, occ_fn, vis_fp

    # Missed Detection
    if N > 0 and M == 0:
        for gt in gt_objects:
            if len(gt.get("bbox", [])) == 4:
                scene_ious.append(0.0)
            else:
                occ_fn += 1
        return scene_ious, scene_maes_vis, scene_maes_occ, occ_tp, occ_fp, occ_fn, vis_fp

    # Empty scene
    if N == 0 and M == 0:
        return [], [], [], 0, 0, 0, 0

    # Cost Matrix
    iou_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            gt_box = gt_objects[i].get("bbox", [])
            pred_box = pred_objects[j].get("bbox", [])
            iou_matrix[i, j] = calculate_iou(gt_box, pred_box)
    
    cost_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            gt_box = gt_objects[i].get("bbox", [])
            pred_box = pred_objects[j].get("bbox", [])

            gt_is_visible = len(gt_box) == 4
            pred_is_visible = len(pred_box) == 4

            classes_match = gt_objects[i].get("type") == pred_objects[j].get("type")
            type_penalty = 0.0 if classes_match else 100.0

            if gt_is_visible and pred_is_visible:
                if iou_matrix[i, j] <= 0.01:
                    cost_matrix[i, j] = 100.0
                else:
                    cost_matrix[i, j] = (1.0 - iou_matrix[i, j]) + type_penalty
            elif not gt_is_visible and not pred_is_visible: # FIX: Both must be occluded
                gt_dist = gt_objects[i].get("distance_m", gt_objects[i].get("distance", 0.0))
                pred_dist = pred_objects[j].get("distance_m", pred_objects[j].get("distance", 0.0))
                dist_cost = min(abs(gt_dist - pred_dist), 20.0)
                cost_matrix[i, j] = dist_cost + type_penalty
            else:
                cost_matrix[i, j] = 100.0 # FIX: Visible cannot match Occluded

    # Matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_gt = set()
    matched_pred = set()

    # Process optimal pairs
    for gt_idx, pred_idx in zip(row_ind, col_ind):
        gt_box = gt_objects[gt_idx].get("bbox", [])
        pred_box = pred_objects[pred_idx].get("bbox", [])
        gt_is_visible = len(gt_box) == 4
        pred_is_visible = len(pred_box) == 4
        
        gt_dist = gt_objects[gt_idx].get("distance_m", gt_objects[gt_idx].get("distance", 0.0))
        pred_dist = pred_objects[pred_idx].get("distance_m", pred_objects[pred_idx].get("distance", 0.0))
        
        if cost_matrix[gt_idx, pred_idx] >= 100.0:
            # FIX: Removed the double-counting of occ_fn and occ_fp here.
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        
        if gt_is_visible and pred_is_visible:
            scene_ious.append(iou_matrix[gt_idx, pred_idx])
            scene_maes_vis.append(abs(gt_dist - pred_dist))
        elif not gt_is_visible and not pred_is_visible:
            dist_error = abs(gt_dist - pred_dist)
            scene_maes_occ.append(dist_error)
            occ_tp += 1

    # Penalize missed objects
    for i in range(N):
        if i not in matched_gt:
            if len(gt_objects[i].get("bbox", [])) == 4:
                scene_ious.append(0.0)
            else:
                occ_fn += 1

    for j in range(M):
        if j not in matched_pred:
            if len(pred_objects[j].get("bbox", [])) == 4:
                scene_ious.append(0.0)
                vis_fp += 1
            else:
                occ_fp += 1

    return scene_ious, scene_maes_vis, scene_maes_occ, occ_tp, occ_fp, occ_fn, vis_fp

def compute_metrics(preds, labels):
    '''Evaluate custom metrics'''
    # Unpack and convert logits to token IDs
    preds, labels = preds, labels

    valid_tasks = list(VALID_TASK_TYPES) + ["all"]
    buckets = {
        tt: {"iou": [], "mae_vis": [], "mae_occ": [], "occ_tp": 0, "occ_fp": 0, "occ_fn": 0, "vis_fp": 0, "true_dec": [], "pred_dec": [], "gt_rat": [], "pred_rat": []}
        for tt in valid_tasks
    }

    for pred_text, label_text in zip(preds, labels):
        pred_json = extract_json_from_text(pred_text)
        label_json = extract_json_from_text(label_text)

        if not isinstance(pred_json, dict):
            pred_json = {}
        if not label_json:
            continue

        # Determine buckets
        task_type = label_json.get("task_type", "unknown").lower()
        target_buckets = ["all"]
        if task_type in VALID_TASK_TYPES:
            target_buckets.append(task_type)

        # Extract data
        gt_decision = normalize_decision(label_json.get("decision"))
        pred_decision = normalize_decision(pred_json.get("decision") if pred_json else None)
        
        gt_rat = extract_rationale_from_text(label_text)
        pred_rat = extract_rationale_from_text(pred_text)

        gt_objects = label_json.get("grounded_objects", [])
        pred_objects = pred_json.get("grounded_objects", []) if pred_json else []

        scene_ious, scene_maes_vis, scene_maes_occ, occ_tp, occ_fp, occ_fn, vis_fp = get_optimal_matches(gt_objects, pred_objects)

        # Append to relevant buckets
        for b in target_buckets:
            buckets[b]["true_dec"].append(gt_decision)
            buckets[b]["pred_dec"].append(pred_decision)
            buckets[b]["gt_rat"].append(gt_rat)
            buckets[b]["pred_rat"].append(pred_rat)
            buckets[b]["iou"].extend(scene_ious)
            buckets[b]["mae_vis"].extend(scene_maes_vis)
            buckets[b]["mae_occ"].extend(scene_maes_occ)
            buckets[b]["occ_tp"] += occ_tp
            buckets[b]["occ_fp"] += occ_fp
            buckets[b]["occ_fn"] += occ_fn
            buckets[b]["vis_fp"] += vis_fp

    final_results = {}

    for b, data in buckets.items():
        if not data["true_dec"]:
            continue
            
        Visible_mIoU = np.mean(data["iou"]) if data["iou"] else 0.0
        MAE_Vis = np.mean(data["mae_vis"]) if data["mae_vis"] else float('nan')
        MAE_Occ = np.mean(data["mae_occ"]) if data["mae_occ"] else float('nan')
        F1 = f1_score(data["true_dec"], data["pred_dec"], labels=VALID_DECISIONS, average='macro', zero_division=0)

        occ_tp = data["occ_tp"]
        occ_fp = data["occ_fp"]
        occ_fn = data["occ_fn"]
        total_occluded_gt = occ_tp + occ_fn
        Occ_Recall = occ_tp / (occ_tp + occ_fn) if (occ_tp + occ_fn) > 0 else 0.0

        tp_under_10m = sum(1 for error in data["mae_occ"] if error <= 10.0)
        tp_under_20m = sum(1 for error in data["mae_occ"] if error <= 20.0)
        tp_under_30m = sum(1 for error in data["mae_occ"] if error <= 30.0)

        Occ_Recall_10m = tp_under_10m / total_occluded_gt if total_occluded_gt > 0 else 0.0
        Occ_Recall_20m = tp_under_20m / total_occluded_gt if total_occluded_gt > 0 else 0.0
        Occ_Recall_30m = tp_under_30m / total_occluded_gt if total_occluded_gt > 0 else 0.0

        bert_f1_mean = 0.0
        if data["pred_rat"] and data["gt_rat"]:
            _, _, B_F1_Tensor = calc_bert_score(
                data["pred_rat"], 
                data["gt_rat"], 
                lang="en", 
                verbose=False, 
                model_type="distilbert-base-uncased"
            )
            
            b_f1_scores = B_F1_Tensor.numpy()
            for i, rat in enumerate(data["pred_rat"]):
                if rat == "No rationale provided.":
                    b_f1_scores[i] = 0.0
            
            bert_f1_mean = np.mean(b_f1_scores)

        # Save
        final_results[f"eval_{b}_Visible_mIoU"] = round(float(Visible_mIoU), 4)
        final_results[f"eval_{b}_Occlusion_Recall"] = round(float(Occ_Recall), 4)
        final_results[f"eval_{b}_Occ_Recall_@10m"] = round(float(Occ_Recall_10m), 4)
        final_results[f"eval_{b}_Occ_Recall_@20m"] = round(float(Occ_Recall_20m), 4)
        final_results[f"eval_{b}_Occ_Recall_@30m"] = round(float(Occ_Recall_30m), 4)
        final_results[f"eval_{b}_Visible_MAE"] = round(float(MAE_Vis), 4)
        final_results[f"eval_{b}_F1"] = round(float(F1), 4)
        final_results[f"eval_{b}_Rationale_BERTScore"] = round(float(bert_f1_mean), 4)

    return final_results