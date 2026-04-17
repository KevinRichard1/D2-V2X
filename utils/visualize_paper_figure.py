import argparse
import json
import math
import os
import re
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.patheffects import withStroke
from PIL import Image

# ── BEV geometry ──────────────────────────────────────────────────────────────
RES    = 0.2
CROP_M = 100.0
IMG_PX = 1000

def world_to_px(x, y):
    return (CROP_M - y) / RES, (CROP_M - x) / RES   # (col, row)

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = "#0a0a14"
COL_HIDDEN  = "#FF4655"
COL_VISIBLE = "#2ECC71"
COL_SENSOR  = "#FFD700"
COL_EGO     = "#00CFFF"
COL_ARROW   = "#FFAA00"
COL_EGO_BG  = "#1a0f00"
COL_EGO_TXT = "#FF9933"
COL_D2_BG   = "#0a1a0a"
COL_D2_TXT  = "#55EE88"

TYPE_DIMS = {
    "car": (4.5, 2.0), "truck": (8.0, 2.8), "trailer": (12.0, 3.0),
    "van": (5.5, 2.3), "bus": (12.0, 3.0),  "pedestrian": (0.7, 0.7),
    "bicycle": (1.8, 0.6),
}

SAMPLE_ID  = "16980779873717473_1"
BEV_PATH   = "Datasets/images/val/bev/1698077987_572028302_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered_bev.png"
TEST_JSON  = "Datasets/Main Dataset/d2_v2x_test.json"
EGO_JSON   = "Results/Raw Results/ego_results.json"
D2_JSON    = "Results/Raw Results/d2v2x_results.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _outline(lw=2):
    return [withStroke(linewidth=lw, foreground="black")]

def extract_json_block(text):
    m = re.search(r"```json(.*?)```", text, re.DOTALL)
    try: return json.loads(m.group(1)) if m else None
    except: return None

def extract_think(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else text.split("```json")[0].strip()

def parse_coords(text):
    pat = r"(\w+)\s+at\s+x:\s*(-?[\d]+\.?[\d]*),\s*y:\s*(-?[\d]+\.?[\d]*)"
    return [(t.lower(), float(x), float(y))
            for t, x, y in re.findall(pat, text, re.I)]


# ── BEV drawing ───────────────────────────────────────────────────────────────

def draw_distance_rings(ax, rings=(10, 20, 30, 40)):
    ox, oy = world_to_px(0, 0)
    for r in rings:
        r_px = r / RES
        ax.add_patch(plt.Circle((ox, oy), r_px, color="white",
                                fill=False, lw=0.5, ls="--", alpha=0.25, zorder=3))
        ax.text(ox + r_px * 0.707 + 3, oy - r_px * 0.707 - 3,
                f"{r}m", color="white", fontsize=6, alpha=0.45, zorder=4)


def draw_cardinal(ax):
    pad = 26
    mid = IMG_PX / 2
    kw = dict(color="white", fontsize=8, fontweight="bold", alpha=0.45,
              path_effects=_outline(1.5), zorder=12)
    ax.text(mid, pad,          "N", ha="center", va="top",    **kw)
    ax.text(mid, IMG_PX - pad, "S", ha="center", va="bottom", **kw)
    ax.text(pad, mid,          "W", ha="left",   va="center", **kw)
    ax.text(IMG_PX - pad, mid, "E", ha="right",  va="center", **kw)


def draw_sensor(ax):
    """Small gold diamond at infrastructure sensor origin — no text label."""
    ox, oy = world_to_px(0, 0)
    ax.plot(ox, oy, "D", color=COL_SENSOR, markersize=9,
            markeredgecolor="white", markeredgewidth=0.8, zorder=11)


def draw_ego_approx(ax):
    """Cyan car marker ~30 m south of sensor (approximate)."""
    ex, ey = world_to_px(-30, 0)
    ax.add_patch(FancyBboxPatch((ex - 12, ey - 20), 24, 40,
                                boxstyle="round,pad=2",
                                ec="white", fc=COL_EGO, lw=1.2, alpha=0.85, zorder=12))
    ax.text(ex, ey, "EGO", color="black", fontsize=7, fontweight="bold",
            ha="center", va="center", zorder=13)


def draw_box(ax, world_x, world_y, otype, dist_m, color,
             lw=2.2, alpha=0.25, zorder=6, fontsize=8, label_pos=None):
    l, w = TYPE_DIMS.get(otype, (4.5, 2.0))
    cx, cy = world_to_px(world_x, world_y)
    hr, hc = (l / 2) / RES, (w / 2) / RES

    ax.add_patch(FancyBboxPatch((cx - hc, cy - hr), 2 * hc, 2 * hr,
                                boxstyle="square,pad=0",
                                ec=color, fc=color, lw=lw, alpha=alpha, zorder=zorder))
    ax.add_patch(FancyBboxPatch((cx - hc, cy - hr), 2 * hc, 2 * hr,
                                boxstyle="square,pad=0",
                                ec=color, fill=False, lw=lw, zorder=zorder + 1))
    ax.plot(cx, cy, "o", color=color, markersize=3,
            markeredgecolor="white", markeredgewidth=0.5, zorder=zorder + 2)

    if label_pos is not None:
        lc, lr = label_pos
        ax.annotate("", xy=(cx, cy), xytext=(lc, lr),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                   mutation_scale=7,
                                   connectionstyle="arc3,rad=0.0"),
                    zorder=zorder + 3)
        ax.text(lc, lr, f"{otype.capitalize()}  {dist_m:.0f} m",
                color="white", fontsize=fontsize, ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none", alpha=0.92),
                path_effects=_outline(), zorder=zorder + 4)
    else:
        ax.text(cx, cy + hr + 5, f"{otype.capitalize()}  {dist_m:.0f} m",
                color="white", fontsize=fontsize, ha="center", va="top",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none", alpha=0.92),
                path_effects=_outline(), zorder=zorder + 3)


def draw_occlusion_shadow(ax, occ_objs):
    if not occ_objs: return
    ox, oy = world_to_px(0, 0)
    cx_w = np.mean([o["x"] for o in occ_objs])
    cy_w = np.mean([o["y"] for o in occ_objs])
    cx_px, cy_px = world_to_px(cx_w, cy_w)
    angle = math.atan2(cy_px - oy, cx_px - ox)
    half  = math.radians(22)
    reach = IMG_PX * 0.7
    pts = [(ox, oy)]
    for i in range(31):
        a = (angle - half) + i * (2 * half / 30)
        pts.append((ox + reach * math.cos(a), oy + reach * math.sin(a)))
    pts.append((ox, oy))
    ax.add_patch(Polygon(pts, closed=True, fc=COL_HIDDEN, ec="none", alpha=0.07, zorder=2))


def annotate_bev(ax, gj, gc, question=""):
    go      = gj.get("grounded_objects", [])
    occ_raw = [o for o in go if not o.get("bbox")]
    vis_raw = [o for o in go if o.get("bbox")]

    # Match rationale coords → hidden objects by distance proximity
    coord_sorted = sorted([(t, x, y, math.sqrt(x**2 + y**2)) for t, x, y in gc],
                          key=lambda c: c[3])
    used = set()
    occ_objs = []
    for obj in occ_raw:
        t, dist = obj.get("type", "car"), obj.get("distance_m", 15)
        best_c, best_err, best_i = None, float("inf"), None
        for i, (ct, cx, cy, cd) in enumerate(coord_sorted):
            if i in used: continue
            if ct != t: continue
            err = abs(cd - dist)
            if err < best_err:
                best_err, best_c, best_i = err, (cx, cy), i
        if best_c and best_err < 12:
            x, y = best_c
            used.add(best_i)
        else:
            x, y = dist, 0
        occ_objs.append({"type": t, "x": x, "y": y, "dist": dist})

    vis_objs = [{"type": o.get("type", "car"), "x": o.get("distance_m", 20),
                 "y": 0, "dist": o.get("distance_m", 0)} for o in vis_raw]

    draw_distance_rings(ax)
    draw_cardinal(ax)
    draw_occlusion_shadow(ax, occ_objs)

    # Visible objects — muted
    for obj in vis_objs:
        draw_box(ax, obj["x"], obj["y"], obj["type"], obj["dist"],
                 color=COL_VISIBLE, lw=1.2, alpha=0.15, zorder=5)

    # Hidden objects — leader-line labels spread left of cluster
    occ_sorted_px = sorted(occ_objs, key=lambda o: world_to_px(o["x"], o["y"])[1])
    n = len(occ_sorted_px)
    if n:
        spacing = 46
        min_col = min(world_to_px(o["x"], o["y"])[0] for o in occ_sorted_px)
        label_col = max(min_col - 140, 40)
        mid_row   = np.mean([world_to_px(o["x"], o["y"])[1] for o in occ_sorted_px])
        total_h   = (n - 1) * spacing
        label_rows = [mid_row - total_h / 2 + i * spacing for i in range(n)]
        for obj, lrow in zip(occ_sorted_px, label_rows):
            draw_box(ax, obj["x"], obj["y"], obj["type"], obj["dist"],
                     color=COL_HIDDEN, lw=2.5, alpha=0.30, zorder=7,
                     label_pos=(label_col, lrow))

    # "In path" arrow
    q_low = question.lower()
    ego_c, ego_r = world_to_px(-30, 0)
    if "left" in q_low:
        dest_c, dest_r, rad = 310, 460, -0.35
        arrow_lbl, al_c, al_r = "Intended\nleft turn", 300, 550
    elif "right" in q_low:
        dest_c, dest_r, rad = 700, 460, 0.35
        arrow_lbl, al_c, al_r = "Intended\nright turn", 710, 550
    else:
        dest_c, dest_r, rad = ego_c - 15, 400, -0.15
        arrow_lbl, al_c, al_r = "Intended\npath", ego_c - 80, (ego_r + 400) / 2

    ax.annotate("", xy=(dest_c, dest_r), xytext=(ego_c, ego_r),
                arrowprops=dict(arrowstyle="-|>", color=COL_ARROW, lw=2.5,
                                mutation_scale=14,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=14)
    ax.text(al_c, al_r, arrow_lbl, color=COL_ARROW, fontsize=7.5,
            ha="center", va="center", fontweight="bold",
            path_effects=_outline(), zorder=14)

    # Conflict label
    if occ_objs:
        cx_px = np.mean([world_to_px(o["x"], o["y"])[0] for o in occ_objs])
        cy_px = np.mean([world_to_px(o["x"], o["y"])[1] for o in occ_objs])
        side  = 1 if cx_px < IMG_PX * 0.65 else -1
        lx = cx_px + side * 170
        ax.annotate("", xy=(cx_px + side * 60, cy_px), xytext=(lx, cy_px),
                    arrowprops=dict(arrowstyle="-|>", color="#FF8888", lw=1.6,
                                   mutation_scale=9), zorder=14)
        ax.text(lx + side * 6, cy_px,
                "In path\n(hidden by occluder)",
                color="#FF8888", fontsize=7.5,
                ha="left" if side > 0 else "right", va="center",
                fontweight="bold", path_effects=_outline(), zorder=14)

    draw_sensor(ax)
    draw_ego_approx(ax)

    # BEV legend
    handles = [
        mpatches.Patch(fc=COL_SENSOR, ec="white", lw=0.8, label="Infrastructure sensor"),
        mpatches.Patch(fc=COL_EGO,    ec="white", lw=0.8, label="Ego vehicle (approx.)"),
        mpatches.Patch(fc=COL_ARROW,  ec="white", lw=0.8, label="Intended maneuver"),
        mpatches.Patch(fc=COL_HIDDEN, ec="white", lw=0.8,
                       label=f"Hidden  ×{len(occ_objs)}  (LiDAR only)"),
    ]
    if vis_objs:
        handles.append(mpatches.Patch(fc=COL_VISIBLE, ec="white", lw=0.8,
                                      label=f"Visible  ×{len(vis_objs)}"))
    ax.legend(handles=handles, loc="lower right", fontsize=7.5,
              framealpha=0.80, facecolor="#0d0d22",
              labelcolor="white", edgecolor="#334466")

    return occ_objs


# ── Camera panel ──────────────────────────────────────────────────────────────

CAM_NAMES = {
    "s110_camera_basler_south1_8mm": "South 1",
    "s110_camera_basler_south2_8mm": "South 2",
    "s110_camera_basler_north_8mm":  "North",
    "s110_camera_basler_east_8mm":   "East",
    "vehicle_camera_basler_16mm":    "Vehicle",
}

def cam_label(path):
    for key, lbl in CAM_NAMES.items():
        if key in path: return lbl
    return "Camera"


def show_cam(ax, path, label):
    ax.set_facecolor("#080818")
    if os.path.exists(path):
        ax.imshow(np.array(Image.open(path).convert("RGB")), aspect="auto")
    else:
        ax.text(0.5, 0.5, "Missing", color="white",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#334466"); sp.set_linewidth(1.2)
    ax.set_title(label, fontsize=7.5, color="white", pad=3, fontweight="bold",
                 bbox=dict(fc="#0d0d22", ec="none", alpha=0.7, pad=1.5))


# ── Answer text box ───────────────────────────────────────────────────────────

SYNONYMS = {
    "stop": "yield", "wait": "yield", "halt": "yield", "brake": "yield",
    "do not proceed": "yield", "not clear": "yield", "not_safe": "yield",
    "proceed": "monitor", "continue": "monitor", "go": "monitor",
    "caution": "monitor", "maintain": "monitor",
    "clear": "safe", "no_hazard": "safe", "yes": "safe",
    "hidden_vehicles": "unsafe", "occluded": "unsafe", "no": "unsafe",
    # danger synonyms — yield/unsafe both mean "don't go"
    "unsafe": "yield",
}

DANGER_DECISIONS = {"yield", "unsafe"}

def normalize_dec(raw):
    if not raw: return None
    r = str(raw).strip().lower()
    return SYNONYMS.get(r, r)

def is_correct(decision, gt_decision):
    nd = normalize_dec(decision)
    ng = normalize_dec(gt_decision)
    # If GT is a danger signal, any danger signal from the model counts
    if ng in DANGER_DECISIONS:
        return nd in DANGER_DECISIONS
    return nd == ng


def show_answer(ax, title, text, bg_color, txt_color, decision, gt_decision):
    correct = is_correct(decision, gt_decision)
    ax.set_facecolor(bg_color)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(txt_color); sp.set_linewidth(1.5)

    dec_color = COL_HIDDEN if not correct else COL_VISIBLE
    ax.text(0.02, 0.94, title, transform=ax.transAxes,
            color=txt_color, fontsize=10, fontweight="bold", va="top")
    dec_lbl = f"  Decision: {str(decision).upper()}"
    dec_mark = "  ✗  missed hazard" if not correct else "  ✓  hazard detected"
    ax.text(0.98, 0.94, dec_lbl + dec_mark, transform=ax.transAxes,
            color=dec_color, fontsize=8.5, fontweight="bold", va="top", ha="right")

    ax.plot([0.01, 0.99], [0.82, 0.82], color=txt_color, linewidth=0.6,
            alpha=0.4, transform=ax.transAxes, clip_on=False)

    wrapped = textwrap.fill(text, width=80)
    ax.text(0.02, 0.76, wrapped, transform=ax.transAxes,
            color="white", fontsize=8, va="top", linespacing=1.4,
            wrap=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="Results/figures/paper_figure.png")
    args = parser.parse_args()

    # Load data
    with open(TEST_JSON) as f:
        test_data = json.load(f)
    with open(EGO_JSON) as f:
        ego_results = {str(item["id"]): item for item in json.load(f)}
    with open(D2_JSON) as f:
        d2_results = {str(item["id"]): item for item in json.load(f)}

    test_item = next(s for s in test_data if str(s["id"]) == SAMPLE_ID)
    ego_item  = ego_results[SAMPLE_ID]
    d2_item   = d2_results[SAMPLE_ID]

    gt_text  = d2_item["ground_truth"]
    gj       = extract_json_block(gt_text)
    gc       = parse_coords(extract_think(gt_text))
    question = re.sub(r"(Image \d+: <image>|BEV: <image>|LiDAR: <lidar>)\n?",
                      "", test_item["conversations"][0]["value"]).strip()

    # Camera paths
    cam_paths = [p.lstrip("./") for p in test_item["file_metadata"]["image_paths"]]

    # Answer texts — extract the human-facing sentence (after </think>, before JSON)
    def clean_answer(text, max_chars=300):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = text.split("```json")[0].strip()
        return text[:max_chars]

    def get_decision(text):
        j = extract_json_block(text)
        return j.get("decision", "?") if j else "?"

    ego_text = clean_answer(ego_item["prediction"], 280)
    d2_text  = clean_answer(d2_item["prediction"], 280)
    ego_dec  = get_decision(ego_item["prediction"])
    d2_dec   = get_decision(d2_item["prediction"])
    gt_dec   = gj.get("decision", "?") if gj else "?"

    # ── Layout ────────────────────────────────────────────────────────────────
    n_cams = len(cam_paths)
    fig = plt.figure(figsize=(20, 13), facecolor=BG)

    gs = gridspec.GridSpec(
        3, n_cams,
        figure=fig,
        height_ratios=[1.8, 5.5, 2.2],
        hspace=0.06, wspace=0.04,
        left=0.02, right=0.98,
        top=0.91, bottom=0.02,
    )

    # Row 0: camera thumbnails
    for i, path in enumerate(cam_paths):
        ax = fig.add_subplot(gs[0, i])
        show_cam(ax, path, cam_label(path))

    # Row 1: BEV (spans all columns)
    ax_bev = fig.add_subplot(gs[1, :])
    ax_bev.set_facecolor("black")

    if os.path.exists(BEV_PATH):
        bev_img = np.array(Image.open(BEV_PATH).convert("RGB"))
        ax_bev.imshow(bev_img, origin="upper", zorder=1)
    else:
        ax_bev.set_facecolor("#050510")
        ax_bev.text(0.5, 0.5, "BEV not found", color="white",
                    ha="center", va="center", transform=ax_bev.transAxes)

    ax_bev.set_xlim(0, IMG_PX)
    ax_bev.set_ylim(IMG_PX, 0)
    ax_bev.set_xticks([]); ax_bev.set_yticks([])
    for sp in ax_bev.spines.values():
        sp.set_edgecolor("#334466"); sp.set_linewidth(1.2)

    if gj:
        annotate_bev(ax_bev, gj, gc, question=question)

    # Row 2: ZS + D2V2X answers (each spans half)
    half = n_cams // 2
    ax_zs = fig.add_subplot(gs[2, :half])
    ax_d2 = fig.add_subplot(gs[2, half:])

    show_answer(ax_zs, "Ego-Only  (vehicle camera + vehicle LiDAR)", ego_text,
                COL_EGO_BG, COL_EGO_TXT, ego_dec, gt_dec)
    show_answer(ax_d2, "D2-V2X  (all cameras + infrastructure LiDAR)", d2_text,
                COL_D2_BG, COL_D2_TXT, d2_dec, gt_dec)

    # ── Question at top ───────────────────────────────────────────────────────
    fig.text(0.5, 0.955,
             f'"{question}"',
             ha="center", va="top",
             fontsize=13, color="white", style="italic",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc="#0d0d22",
                       ec="#334466", alpha=0.9))

    # ── Sensor legend (inline in title area) ─────────────────────────────────
    fig.text(0.01, 0.955,
             "D2-V2X  ·  V2X Cooperative Perception",
             ha="left", va="top", fontsize=10, color="#8899bb")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
