"""Evaluate model predictions against ground truth.

Parses model output JSON blocks, normalizes decisions, and reports
accuracy broken down by task type (maneuver / counting / spatial).

Usage:
    python utils/evaluate.py \
        --predictions results/zero_shot_results.json \
        --ground_truth "Datasets/Main Dataset/d2_v2x_test.json"

    # Compare two runs side-by-side:
    python utils/evaluate.py \
        --predictions results/finetuned_results.json \
        --ground_truth "Datasets/Main Dataset/d2_v2x_test.json" \
        --baseline results/zero_shot_results.json
"""

import argparse
import json
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Valid label sets (from ground truth)
# ---------------------------------------------------------------------------
VALID_DECISIONS   = {"monitor", "yield", "safe", "unsafe"}
VALID_TASK_TYPES  = {"maneuver", "counting", "spatial"}
VALID_HAZARD      = {"high", "medium", "low", "none"}

# Synonym → canonical decision.  Covers common zero-shot drifts.
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


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_json_block(text: str) -> dict | None:
    '''Extract the first ```json ... ``` block from model output.'''
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def normalize_decision(raw: str | None) -> str | None:
    '''Map a raw decision string to one of the 4 canonical values, or None.'''
    if raw is None:
        return None
    raw = str(raw).strip().lower()
    if raw in VALID_DECISIONS:
        return raw
    return DECISION_SYNONYMS.get(raw)


def normalize_hazard(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = str(raw).strip().lower()
    return raw if raw in VALID_HAZARD else None


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(path: str) -> dict:
    '''Returns {sample_id: {task_type, decision, hazard_level, count}}.'''
    with open(path) as f:
        data = json.load(f)

    gt = {}
    for item in data:
        sid = item['id']
        for conv in item.get('conversations', []):
            if conv['from'] == 'assistant':
                block = extract_json_block(conv['value'])
                if block:
                    gt[sid] = {
                        'task_type':   block.get('task_type'),
                        'decision':    block.get('decision'),
                        'hazard_level': block.get('hazard_level'),
                        'count':       block.get('count'),
                    }
                break
    return gt


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(predictions: dict, ground_truth: dict) -> dict:
    '''
    predictions : {sample_id: raw_response_string}
    ground_truth: {sample_id: {task_type, decision, hazard_level, count}}
    Returns a nested metrics dict.
    '''
    # Counters per task type + global
    task_types = list(VALID_TASK_TYPES) + ['all']

    stats = {tt: defaultdict(int) for tt in task_types}

    for sid, gt in ground_truth.items():
        tt = gt.get('task_type') or 'unknown'
        buckets = [tt, 'all'] if tt in VALID_TASK_TYPES else ['all']

        raw_response = predictions.get(sid)

        for b in buckets:
            stats[b]['total'] += 1

            if raw_response is None:
                stats[b]['missing_prediction'] += 1
                continue

            block = extract_json_block(raw_response)

            if block is None:
                stats[b]['no_json_block'] += 1
                continue

            stats[b]['has_json'] += 1

            # task_type
            pred_tt = str(block.get('task_type', '')).strip().lower()
            if pred_tt == tt:
                stats[b]['task_type_correct'] += 1

            # decision (exact)
            pred_dec_raw = block.get('decision')
            gt_dec = gt.get('decision')
            if pred_dec_raw is not None and str(pred_dec_raw).strip().lower() == str(gt_dec).strip().lower():
                stats[b]['decision_exact'] += 1

            # decision (normalized)
            pred_dec_norm = normalize_decision(pred_dec_raw)
            if pred_dec_norm == gt_dec:
                stats[b]['decision_norm'] += 1

            # hazard_level
            pred_hl = normalize_hazard(block.get('hazard_level'))
            gt_hl = gt.get('hazard_level')
            if pred_hl == gt_hl:
                stats[b]['hazard_correct'] += 1

            # count (counting tasks only)
            if tt == 'counting':
                gt_count = gt.get('count')
                pred_count = block.get('count')
                if gt_count is not None and pred_count is not None:
                    try:
                        pred_count_int = int(pred_count)
                        stats[b]['count_total'] += 1
                        if pred_count_int == int(gt_count):
                            stats[b]['count_exact'] += 1
                        if abs(pred_count_int - int(gt_count)) <= 1:
                            stats[b]['count_off_by_one'] += 1
                    except (ValueError, TypeError):
                        stats[b]['count_parse_error'] += 1

    return {tt: dict(s) for tt, s in stats.items()}


def pct(num, denom):
    return f'{100 * num / denom:.1f}%' if denom > 0 else 'N/A'


def print_report(stats: dict, label: str = ''):
    header = f'  Evaluation Report{" — " + label if label else ""}  '
    print()
    print('=' * (len(header) + 4))
    print(f'  {header}')
    print('=' * (len(header) + 4))

    col_order = ['maneuver', 'counting', 'spatial', 'all']
    col_labels = ['Maneuver', 'Counting', 'Spatial', 'TOTAL']

    rows = [
        ('Samples',          lambda s: str(s.get('total', 0))),
        ('JSON parsed',      lambda s: f"{s.get('has_json', 0)} ({pct(s.get('has_json',0), s.get('total',0))})"),
        ('No JSON block',    lambda s: f"{s.get('no_json_block', 0)} ({pct(s.get('no_json_block',0), s.get('total',0))})"),
        ('Missing pred',     lambda s: f"{s.get('missing_prediction', 0)}"),
        ('─' * 22,           lambda s: '─' * 14),
        ('Decision (exact)', lambda s: pct(s.get('decision_exact', 0), s.get('total', 0))),
        ('Decision (norm)',  lambda s: pct(s.get('decision_norm',  0), s.get('total', 0))),
        ('Hazard level',     lambda s: pct(s.get('hazard_correct', 0), s.get('total', 0))),
        ('Task type output', lambda s: pct(s.get('task_type_correct', 0), s.get('has_json', 0))),
        ('─' * 22,           lambda s: '─' * 14),
        ('Count exact',      lambda s: pct(s.get('count_exact', 0),       s.get('count_total', 0)) if s.get('count_total') else 'N/A'),
        ('Count ±1',         lambda s: pct(s.get('count_off_by_one', 0),  s.get('count_total', 0)) if s.get('count_total') else 'N/A'),
    ]

    col_w = 16
    hdr = f"{'Metric':<24}" + ''.join(f'{lbl:>{col_w}}' for lbl in col_labels)
    print()
    print(hdr)
    print('-' * len(hdr))

    for row_label, fn in rows:
        if row_label.startswith('─'):
            print('-' * len(hdr))
            continue
        cells = [fn(stats.get(tt, {})) for tt in col_order]
        print(f'{row_label:<24}' + ''.join(f'{c:>{col_w}}' for c in cells))

    print()


def compare_reports(baseline_stats: dict, pred_stats: dict,
                    baseline_label: str, pred_label: str):
    '''Print side-by-side delta table for two runs.'''
    metrics = [
        ('Decision (norm)',  lambda s: s.get('decision_norm',  0) / max(s.get('total', 1), 1)),
        ('Hazard level',     lambda s: s.get('hazard_correct', 0) / max(s.get('total', 1), 1)),
        ('JSON parsed',      lambda s: s.get('has_json',       0) / max(s.get('total', 1), 1)),
        ('Count exact',      lambda s: s.get('count_exact',    0) / max(s.get('count_total', 1), 1) if s.get('count_total') else None),
    ]

    print()
    print('=' * 72)
    print(f'  Comparison: {baseline_label}  →  {pred_label}')
    print('=' * 72)

    col_order = ['maneuver', 'counting', 'spatial', 'all']
    col_labels = ['Maneuver', 'Counting', 'Spatial', 'TOTAL']
    col_w = 18

    hdr = f"{'Metric':<22}" + ''.join(f'{lbl:>{col_w}}' for lbl in col_labels)
    print()
    print(hdr)
    print('-' * len(hdr))

    for metric_label, fn in metrics:
        cells = []
        for tt in col_order:
            b_val = fn(baseline_stats.get(tt, {}))
            p_val = fn(pred_stats.get(tt, {}))
            if b_val is None or p_val is None:
                cells.append('N/A')
            else:
                delta = (p_val - b_val) * 100
                sign = '+' if delta >= 0 else ''
                cells.append(f'{p_val*100:.1f}% ({sign}{delta:.1f}pp)')
        print(f'{metric_label:<22}' + ''.join(f'{c:>{col_w}}' for c in cells))

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate D2-V2X model predictions')
    parser.add_argument('--predictions',  required=True,
                        help='Path to predictions JSON {sample_id: response_str}')
    parser.add_argument('--ground_truth', required=True,
                        help='Path to ground truth dataset JSON (d2_v2x_*.json)')
    parser.add_argument('--baseline',     default=None,
                        help='Optional baseline predictions for delta comparison')
    parser.add_argument('--label',        default='',
                        help='Label for this run (shown in report header)')
    parser.add_argument('--output',       default=None,
                        help='Optional path to save metrics as JSON')
    args = parser.parse_args()

    print(f'Loading ground truth: {args.ground_truth}')
    gt = load_ground_truth(args.ground_truth)
    print(f'  {len(gt)} samples loaded')

    print(f'Loading predictions:  {args.predictions}')
    with open(args.predictions) as f:
        preds = json.load(f)
    print(f'  {len(preds)} predictions loaded')

    stats = compute_metrics(preds, gt)
    print_report(stats, label=args.label or args.predictions)

    if args.baseline:
        print(f'Loading baseline:     {args.baseline}')
        with open(args.baseline) as f:
            baseline_preds = json.load(f)
        baseline_stats = compute_metrics(baseline_preds, gt)
        compare_reports(baseline_stats, stats,
                        baseline_label=args.baseline,
                        pred_label=args.predictions)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'Metrics saved to {args.output}')


if __name__ == '__main__':
    main()
