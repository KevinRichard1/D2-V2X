'''Module to calculate metrics'''
import re
import json
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
from bert_score import score as calc_bert_score

VALID_TASK_TYPES  = {"maneuver", "counting", "spatial"}

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
    parts = text.split("```json")
    if len(parts) > 0:
        rationale = parts[0].strip()
        return rationale if rationale else "No rationale provided."
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

def get_optimal_matches(gt_objects, pred_objects, penalty_dist=100.0):
    '''Pair Predicted boxes to ground truth boxes'''
    N = len(gt_objects)
    M = len(pred_objects)

    scene_ious = []
    scene_maes = []

    # Hallucination
    if N == 0 and M > 0:
        return [0.0] * M, []

    # Missed Detection
    if N > 0 and M == 0:
        return [0.0] * N, [penalty_dist] * N

    # Empty scene
    if N == 0 and M == 0:
        return [], []

    # Cost Matrix
    iou_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            gt_box = gt_objects[i].get("bbox", [0, 0, 0, 0])
            pred_box = pred_objects[j].get("bbox", [0, 0, 0, 0])
            iou_matrix[i, j] = calculate_iou(gt_box, pred_box)

    cost_matrix = 1.0 - iou_matrix

    # Matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_gt = set()
    matched_pred = set()

    # Process optimal pairs
    for gt_idx, pred_idx in zip(row_ind, col_ind):
        iou = iou_matrix[gt_idx, pred_idx]
        scene_ious.append(iou)

        if iou > 0:
            gt_dist = gt_objects[gt_idx].get("distance_m", 0.0)
            pred_dist = pred_objects[pred_idx].get("distance_m", 0.0)
            scene_maes.append(abs(gt_dist - pred_dist))
        else:
            scene_maes.append(penalty_dist)

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

    # Penalize missed objects
    for i in range(N):
        if i not in matched_gt:
            scene_ious.append(0.0)
            scene_maes.append(penalty_dist)

    # Penalize hallucinations
    for j in range(M):
        if j not in matched_pred:
            scene_ious.append(0.0)

    return scene_ious, scene_maes

def compute_metrics(eval_pred, tokenizer):
    '''Evaluate custom metrics'''
    # Unpack and convert logits to token IDs
    preds, labels = eval_pred
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    preds = np.where(preds != -100, preds, pad_id)
    labels = np.where(labels != -100, labels, pad_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    valid_tasks = list(VALID_TASK_TYPES) + ["all"]
    buckets = {
        tt: {"iou": [], "mae": [], "true_dec": [], "pred_dec": [], "gt_rat": [], "pred_rat": []} 
        for tt in valid_tasks
    }

    for pred_text, label_text in zip(decoded_preds, decoded_labels):
        pred_json = extract_json_from_text(pred_text)
        label_json = extract_json_from_text(label_text)

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
        scene_ious, scene_maes = get_optimal_matches(gt_objects, pred_objects, penalty_dist=100.0)

        # Append to relevant buckets
        for b in target_buckets:
            buckets[b]["true_dec"].append(gt_decision)
            buckets[b]["pred_dec"].append(pred_decision)
            buckets[b]["gt_rat"].append(gt_rat)
            buckets[b]["pred_rat"].append(pred_rat)
            buckets[b]["iou"].extend(scene_ious)
            buckets[b]["mae"].extend(scene_maes)

    final_results = {}

    for b, data in buckets.items():
        if not data["true_dec"]:
            continue
            
        mIoU = np.mean(data["iou"]) if data["iou"] else 0.0
        MAE = np.mean(data["mae"]) if data["mae"] else 0.0
        F1 = f1_score(data["true_dec"], data["pred_dec"], average='macro', zero_division=0)

        bert_f1_mean = 0.0
        if data["pred_rat"] and data["gt_rat"]:
            _, _, B_F1 = calc_bert_score(
                data["pred_rat"], 
                data["gt_rat"], 
                lang="en", 
                verbose=False, 
                model_type="distilbert-base-uncased"
            )
            bert_f1_mean = B_F1.mean().item()

        # Save
        final_results[f"eval_{b}_mIoU"] = round(float(mIoU), 4)
        final_results[f"eval_{b}_MAE"] = round(float(MAE), 4)
        final_results[f"eval_{b}_F1"] = round(float(F1), 4)
        final_results[f"eval_{b}_Rationale_BERTScore"] = round(float(bert_f1_mean), 4)

    return final_results