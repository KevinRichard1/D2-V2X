'''Generate Bird's Eye View (BEV) images from LiDAR point clouds.'''
import os
import json
import math
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
METRICS_DIR    = '../data/metrics'
OUTPUT_BASE    = '../data'          # BEV images go to OUTPUT_BASE/{split}/bev/

# ── BEV rasterisation parameters ─────────────────────────────────────────────
X_RANGE   = (-125.0, 125.0)  # metres along X (forward = +X)
Y_RANGE   = (-125.0, 125.0)  # metres along Y (left   = +Y)
Z_MIN     = -3.0              # metres – below this is treated as ground noise
Z_MAX     =  2.5              # metres – colour scale ceiling
RESOLUTION = 0.2              # metres per pixel

IMG_H = int((X_RANGE[1] - X_RANGE[0]) / RESOLUTION)   # 1250 px
IMG_W = int((Y_RANGE[1] - Y_RANGE[0]) / RESOLUTION)   # 1250 px

DEFAULT_COLOR = (1.0, 1.0, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# Rasterisation
# ─────────────────────────────────────────────────────────────────────────────
def rasterize_pcd(points):
    # Spatial Filter
    mask = (points[:, 0] >= X_RANGE[0]) & (points[:, 0] < X_RANGE[1]) & \
           (points[:, 1] >= Y_RANGE[0]) & (points[:, 1] < Y_RANGE[1])
    points = points[mask]
    if points.size == 0: return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    # Map to pixels
    cols = ((Y_RANGE[1] - points[:, 1]) / RESOLUTION).astype(np.int32)
    rows = ((X_RANGE[1] - points[:, 0]) / RESOLUTION).astype(np.int32)
    cols = np.clip(cols, 0, IMG_W - 1)
    rows = np.clip(rows, 0, IMG_H - 1)
    
    # Z-Buffer
    z_grid = np.full((IMG_H, IMG_W), -999.0) 
    indices = np.argsort(points[:, 2])
    z_grid[rows[indices], cols[indices]] = points[indices, 2]

    # Adaptive Binning
    valid_z = z_grid[z_grid > -999.0]
    if valid_z.size == 0: return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    
    ground_ref = np.percentile(valid_z, 15)
    
    # Bins relative to the detected ground
    bins = np.array([ground_ref, ground_ref + 0.3, ground_ref + 1.2, ground_ref + 2.5])
    
    inds = np.digitize(z_grid, bins)

    palette = np.array([
        [0, 0, 0],       # 0: Empty
        [30, 30, 60],    # 1: Road (Steel Blue)
        [75, 0, 130],    # 2: Low/Curbs (Indigo)
        [255, 20, 147],  # 3: Cars (Hot Pink)
        [255, 255, 0]    # 4: Tall/Trucks (Yellow)
    ], dtype=np.uint8)

    inds[z_grid == -999.0] = 0
    return palette[inds]

# ─────────────────────────────────────────────────────────────────────────────
# Per-frame BEV generation
# ─────────────────────────────────────────────────────────────────────────────

from PIL import Image

def generate_bev_image(pcd_rel_path, output_path):
    abs_pcd_path = get_abs_path(pcd_rel_path)
    
    if not os.path.exists(abs_pcd_path):
        if not os.path.exists(pcd_rel_path):
            return False
        abs_pcd_path = pcd_rel_path

    try:
        pcd = o3d.io.read_point_cloud(abs_pcd_path)
        points = np.asarray(pcd.points)
        if points.size == 0:
            return False
    except Exception as e:
        print(f"Error reading {abs_pcd_path}: {e}")
        return False
    
    rgb_array = rasterize_pcd(points)

    c_h, c_w = IMG_H // 2, IMG_W // 2
    crop_size = 500
    crop = rgb_array[c_h - crop_size : c_h + crop_size, 
                     c_w - crop_size : c_w + crop_size]
    
    img = Image.fromarray(crop)
    img.save(output_path)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-split processing
# ─────────────────────────────────────────────────────────────────────────────

def get_abs_path(rel_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_path = rel_path.lstrip('./')
    return os.path.join(base_dir, clean_path)

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
        if 'images' not in frame:
            frame['images'] = {}
        frame['images']['bev_bird_eye_view'] = rel_bev_path

        frame.pop('lidar_files', None)

        if os.path.exists(out_path):
            generated += 1
            continue

        ok = generate_bev_image(pcd_path, out_path)
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
    print(f"  Z colour   : [{Z_MIN}, {Z_MAX}] m")

    for split in ('train', 'val', 'test'):
        metrics_path = os.path.join(METRICS_DIR, f'{split}_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"[SKIP] {metrics_path} not found – run parse_data.py first.")
            continue
        process_split(split, metrics_path)

    print('\nDone.  bev_path fields written to *_bev_metrics.json files (originals unchanged).')
