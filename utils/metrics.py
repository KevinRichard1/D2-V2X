'''Module to calculate metrics'''
import re
import json
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
from bert_score import score as calc_bert_score

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

    gt_rationales = []
    pred_rationales = []
    iou_scores = []
    mae_scores = []
    y_true_decisions = []
    y_pred_decisions = []

    for pred_text, label_text in zip(decoded_preds, decoded_labels):
        pred_json = extract_json_from_text(pred_text)
        label_json = extract_json_from_text(label_text)

        if not label_json:
            continue

        pred_rationales.append(extract_rationale_from_text(pred_text))
        gt_rationales.append(extract_rationale_from_text(label_text))

        # F1 for decisions
        gt_decision = label_json.get("decision", "").lower()
        y_true_decisions.append(gt_decision)

        if pred_json and "decision" in pred_json:
            y_pred_decisions.append(pred_json["decision"].lower())
        else:
            y_pred_decisions.append("invalid_decision")

        # Get objects
        gt_objects = label_json.get("grounded_objects", [])
        pred_objects = pred_json.get("grounded_objects", []) if pred_json else []

        scene_ious, scene_maes = get_optimal_matches(gt_objects, pred_objects, penalty_dist=100.0)

        iou_scores.extend(scene_ious)
        mae_scores.extend(scene_maes)

    bert_f1_mean = 0.0
    if pred_rationales and gt_rationales:
        P, R, F1 = calc_bert_score(
            pred_rationales, 
            gt_rationales, 
            lang="en", 
            verbose=False, 
            model_type="distilbert-base-uncased"
        )
        bert_f1_mean = F1.mean().item()

    mIoU = np.mean(iou_scores) if iou_scores else 0.0
    MAE = np.mean(mae_scores) if mae_scores else 0.0

    if y_true_decisions:
        F1 = f1_score(y_true_decisions, y_pred_decisions, average='macro', zero_division=0)
    else:
        F1 = 0.0

    return {
        "eval_Rationale_BERTScore": round(float(bert_f1_mean), 4),
        "eval_mIoU": round(float(mIoU), 4),
        "eval_MAE": round(float(MAE), 4),
        "eval_F1": round(float(F1), 4)
    }