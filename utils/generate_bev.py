'''Generate Bird's Eye View (BEV) images from LiDAR point clouds.

Reads metrics JSON files produced by parse_data.py, loads the corresponding
registered PCD file for each frame, rasterizes the point cloud into a
height-coloured top-down image, overlays annotated bounding boxes, and saves
the result as a PNG.  The bev_path for each frame is written into a new
*_bev_metrics.json file (leaving the original *_metrics.json intact) so
that validate_qa.py can consume it without risk of data loss on crashes.

Run order:
    1. parse_data.py   -> data/metrics/{split}_metrics.json
    2. generate_bev.py -> data/{split}/bev/*.png  +  bev_path added to metrics
    3. validate_qa.py  -> Datasets/…
'''
import os
import json
import math
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')   # headless rendering
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
METRICS_DIR    = '../data/metrics'
OUTPUT_BASE    = '../data'          # BEV images go to OUTPUT_BASE/{split}/bev/

# ── BEV rasterisation parameters ─────────────────────────────────────────────
X_RANGE   = (-125.0, 125.0)  # metres along X (forward = +X)
Y_RANGE   = (-125.0, 125.0)  # metres along Y (left   = +Y)
Z_MIN     = -3.0              # metres – below this is treated as ground noise
Z_MAX     =  5.0              # metres – colour scale ceiling
RESOLUTION = 0.2              # metres per pixel

IMG_H = int((X_RANGE[1] - X_RANGE[0]) / RESOLUTION)   # 1250 px
IMG_W = int((Y_RANGE[1] - Y_RANGE[0]) / RESOLUTION)   # 1250 px

# ── Object-type colours (RGB 0-1) ─────────────────────────────────────────────
TYPE_COLORS = {
    'car':        (0.25, 0.65, 1.00),
    'truck':      (1.00, 0.40, 0.10),
    'van':        (1.00, 0.78, 0.10),
    'pedestrian': (0.10, 1.00, 0.45),
    'bicycle':    (0.80, 0.10, 1.00),
    'motorcycle': (1.00, 0.10, 0.80),
    'trailer':    (1.00, 0.55, 0.00),
}
DEFAULT_COLOR = (1.0, 1.0, 0.0)

# ── Heading string → yaw in radians (fallback when yaw_rad absent) ────────────
HEADING_TO_YAW = {
    'facing east':     0.0,
    'facing forward':  0.0,
    'facing north':    math.pi / 2,
    'facing left':     math.pi / 2,
    'facing west':     math.pi,
    'facing backward': math.pi,
    'facing south':   -math.pi / 2,
    'facing right':   -math.pi / 2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def world_to_pixel(x, y):
    '''Convert world (x, y) metres → image (col, row) pixels.

    Convention:
        • +X  (forward)  → top  of image  (row decreases)
        • +Y  (left)     → left of image  (col decreases)
    '''
    col = int((Y_RANGE[1] - y) / RESOLUTION)
    row = int((X_RANGE[1] - x) / RESOLUTION)
    return col, row


def box_corners_pixels(x, y, length, width, yaw):
    '''Return the 4 pixel-coordinate corners of an oriented 2-D bounding box.

    Args:
        x, y    : box centre in world metres
        length  : extent along the heading direction (metres)
        width   : extent perpendicular to heading   (metres)
        yaw     : heading angle in radians
    Returns:
        list of (col, row) tuples
    '''
    c, s   = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0,  width  / 2.0

    world_corners = [
        (x + c * hl - s * hw,  y + s * hl + c * hw),   # front-left
        (x + c * hl + s * hw,  y + s * hl - c * hw),   # front-right
        (x - c * hl + s * hw,  y - s * hl - c * hw),   # rear-right
        (x - c * hl - s * hw,  y - s * hl + c * hw),   # rear-left
    ]
    return [world_to_pixel(wx, wy) for wx, wy in world_corners]


# ─────────────────────────────────────────────────────────────────────────────
# Rasterisation
# ─────────────────────────────────────────────────────────────────────────────

def rasterize_pcd(pcd_rel_path):
    '''Load a PCD file and produce a height-coloured (IMG_H × IMG_W × 3) uint8 array.

    Points are coloured by their Z value using the 'plasma' colormap.
    When multiple points project to the same pixel, the highest Z wins.

    Returns None if the file is missing or empty.
    '''
    abs_path = os.path.normpath(os.path.join('../utils', pcd_rel_path))
    if not os.path.exists(abs_path):
        # Also try path relative to project root
        abs_path = pcd_rel_path.replace('./', '../')
    if not os.path.exists(abs_path):
        return None

    pcd    = o3d.io.read_point_cloud(abs_path)
    points = np.asarray(pcd.points)   # (N, 3)

    if len(points) == 0:
        return None

    # ── Spatial filter ────────────────────────────────────────────────────────
    mask = (
        (points[:, 0] >= X_RANGE[0]) & (points[:, 0] < X_RANGE[1]) &
        (points[:, 1] >= Y_RANGE[0]) & (points[:, 1] < Y_RANGE[1])
    )
    points = points[mask]
    if len(points) == 0:
        return None

    # ── Pixel mapping ─────────────────────────────────────────────────────────
    cols = np.clip(((Y_RANGE[1] - points[:, 1]) / RESOLUTION).astype(np.int32), 0, IMG_W - 1)
    rows = np.clip(((X_RANGE[1] - points[:, 0]) / RESOLUTION).astype(np.int32), 0, IMG_H - 1)

    # ── Max-Z rasterisation (sort ascending → later writes win) ───────────────
    z_grid = np.full((IMG_H, IMG_W), np.nan, dtype=np.float32)
    order  = np.argsort(points[:, 2])
    z_grid[rows[order], cols[order]] = points[order, 2]

    # ── Height → colour ───────────────────────────────────────────────────────
    z_norm  = np.clip((z_grid - Z_MIN) / (Z_MAX - Z_MIN), 0.0, 1.0)
    cmap    = plt.get_cmap('plasma')
    rgb     = (cmap(z_norm)[:, :, :3] * 255).astype(np.uint8)

    # Black background for pixels with no points
    no_pts = np.isnan(z_grid)
    rgb[no_pts] = 0

    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Box drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_boxes(ax, objects):
    '''Overlay oriented bounding boxes and distance labels on *ax*.

    Occluded objects are drawn with a dashed border.
    '''
    for obj in objects:
        x        = obj.get('x', 0.0)
        y        = obj.get('y', 0.0)
        length   = obj.get('length_m', 1.0)
        width    = obj.get('width_m',  1.0)
        yaw      = obj.get('yaw_rad',  None)
        heading  = obj.get('heading',  'facing east')
        obj_type = obj.get('type',     'unknown')
        vis      = obj.get('visibility', 'clear')
        dist     = obj.get('distance_m', 0.0)

        # Use stored yaw, fall back to heading string
        if yaw is None:
            yaw = HEADING_TO_YAW.get(heading, 0.0)

        color     = TYPE_COLORS.get(obj_type, DEFAULT_COLOR)
        linestyle = '--' if vis != 'clear' else '-'

        corners = box_corners_pixels(x, y, length, width, yaw)

        # Skip if entirely out of view
        if not any(0 <= c < IMG_W and 0 <= r < IMG_H for c, r in corners):
            continue

        poly = plt.Polygon(corners, closed=True, fill=False,
                           edgecolor=color, linewidth=1.5, linestyle=linestyle)
        ax.add_patch(poly)

        # Label at box centre
        cx, cy = world_to_pixel(x, y)
        if 0 <= cx < IMG_W and 0 <= cy < IMG_H:
            ax.text(cx, cy, f"{obj_type}\n{dist:.0f}m",
                    fontsize=5, color=color, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                              alpha=0.55, edgecolor='none'))


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame BEV generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_bev_image(pcd_path, objects, output_path):
    '''Rasterize one frame and save the result as a PNG.

    Returns True on success, False if the PCD file could not be loaded.
    '''
    rgb = rasterize_pcd(pcd_path)
    if rgb is None:
        return False

    # ── Canvas: exactly IMG_H × IMG_W pixels ─────────────────────────────────
    fig = plt.figure(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    ax  = fig.add_axes([0, 0, 1, 1])   # no margins
    ax.imshow(rgb, origin='upper', extent=[0, IMG_W, IMG_H, 0], aspect='auto')
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.axis('off')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # ── Sensor-origin marker ──────────────────────────────────────────────────
    oc, or_ = world_to_pixel(0, 0)
    ax.plot(oc, or_, 'w+', markersize=10, markeredgewidth=2)

    # ── Annotated boxes ───────────────────────────────────────────────────────
    draw_boxes(ax, objects)

    # ── Scale bar (10 m) ──────────────────────────────────────────────────────
    bar_px  = int(10 / RESOLUTION)    # 10 m in pixels
    bar_x0  = 20
    bar_y   = IMG_H - 20
    ax.plot([bar_x0, bar_x0 + bar_px], [bar_y, bar_y], 'w-', linewidth=2)
    ax.text(bar_x0 + bar_px // 2, bar_y - 8, '10 m',
            color='white', fontsize=6, ha='center')

    # ── Direction label ───────────────────────────────────────────────────────
    ax.text(IMG_W // 2, 12, 'FORWARD  (+X)',
            color='white', fontsize=6, ha='center', va='top')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches=None, pad_inches=0,
                facecolor='black', dpi=100)
    plt.close(fig)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-split processing
# ─────────────────────────────────────────────────────────────────────────────

def process_split(split, metrics_path):
    '''Generate BEV images for every frame in *split* and update metrics JSON.

    Adds a ``bev_path`` field to each frame entry so that validate_qa.py can
    reference the image without re-deriving the filename.
    '''
    with open(metrics_path) as f:
        frames = json.load(f)

    bev_dir = os.path.join(OUTPUT_BASE, split, 'bev')
    os.makedirs(bev_dir, exist_ok=True)

    generated = skipped = 0

    for frame in tqdm(frames, desc=f'{split:5s}', unit='frame'):
        lidar_files = frame.get('lidar_files', [])
        if not lidar_files:
            skipped += 1
            continue

        # The registered (fused) PCD is the last entry
        pcd_path = lidar_files[-1]

        # Derive a stable output filename from the PCD stem
        pcd_stem      = os.path.basename(pcd_path).replace('.pcd', '')
        out_filename  = f"{pcd_stem}_bev.png"
        out_path      = os.path.join(bev_dir, out_filename)
        rel_bev_path  = f"./data/{split}/bev/{out_filename}"

        # Always record the path even if the image already exists
        frame['bev_path'] = rel_bev_path

        if os.path.exists(out_path):
            generated += 1
            continue

        ok = generate_bev_image(pcd_path, frame.get('objects', []), out_path)
        if ok:
            generated += 1
        else:
            skipped += 1
            print(f"  [WARN] Could not load PCD: {pcd_path}")

    # ── Write bev_path into a new file to preserve the original ──────────────
    bev_metrics_path = metrics_path.replace('_metrics.json', '_bev_metrics.json')
    with open(bev_metrics_path, 'w') as f:
        json.dump(frames, f, indent=4)

    print(f"  {split}: generated={generated}  skipped={skipped}")
    print(f"  Saved BEV metrics to {bev_metrics_path}")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("BEV Generation")
    print(f"  Resolution : {RESOLUTION} m/px  →  {IMG_H}×{IMG_W} px")
    print(f"  X range    : {X_RANGE} m")
    print(f"  Y range    : {Y_RANGE} m")
    print(f"  Z colour   : [{Z_MIN}, {Z_MAX}] m  (plasma colormap)\n")

    for split in ('train', 'val', 'test'):
        metrics_path = os.path.join(METRICS_DIR, f'{split}_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"[SKIP] {metrics_path} not found – run parse_data.py first.")
            continue
        process_split(split, metrics_path)

    print('\nDone.  bev_path fields written to *_bev_metrics.json files (originals unchanged).')
