# D2-V2X

**D2-V2X** (Data-Driven Vehicle-to-Everything) is a dataset generation and fine-tuning pipeline for [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), a multi-modal vision-language model. The pipeline ingests raw sensor data from the [TUMTraf A9 dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_4) — fused infrastructure + vehicle LiDAR and five camera streams — and produces structured question-answer datasets that teach the model to reason about occluded hazards and cooperative driving decisions.

---

## How It Works

```
Raw Sensor Data  (cameras + LiDAR)
        │
        ▼
  parse_data.py       ← parse annotations, detect occlusions, project 3-D → 2-D
        │
        ▼
  generate_bev.py     ← rasterize LiDAR point clouds into Bird's Eye View PNGs
        │
        ▼
  generate_qa.py      ← call GPT-4o to generate 10 QA pairs per frame
        │
        ▼
  validate_qa.py      ← validate against ground truth, format for Qwen-VL SFT
        │
        ▼
  Datasets/           ← ready-to-use fine-tuning JSON files
```

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/KevinRichard1/D2-V2X
cd D2-V2X/utils
pip install -r requirements.txt
```

Create a `.env` file inside `utils/`:

```
OPENAI_API_KEY=sk-...
BATCH_JOB_ID=batch-...    # only needed when running validate_qa.py
```

**Data & model weights** are not included in the repo (too large). You will need:
- [TUMTraf A9 dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_4) extracted to `data/`
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) weights placed in `qwen/`
- A high-compute environment (GPU cluster or cloud instance) for fine-tuning runs

Expected `data/` layout after extraction:

```
data/
├── calib/                          # camera calibration JSONs
├── train/
│   ├── images/
│   │   ├── s110_camera_basler_south1_8mm/
│   │   ├── s110_camera_basler_south2_8mm/
│   │   ├── s110_camera_basler_north_8mm/
│   │   ├── s110_camera_basler_east_8mm/
│   │   └── vehicle_camera_basler_16mm/
│   ├── point_clouds/
│   │   └── s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/
│   └── labels_point_clouds/
│       └── s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/
└── val/                            # same structure as train/
```

---

## Running the Pipeline

Run each step from the `utils/` directory in order.

### Step 1 — Parse sensor data

```bash
cd utils
python parse_data.py
```

Reads raw annotation JSONs and PCD files, calculates occlusions and 3-D → 2-D projections, and writes structured per-frame metrics.

- **Input:** `data/{train,val}/labels_point_clouds/`
- **Output:** `data/metrics/{train,val,test}_metrics.json`
- **Config:** Set `DEBUG = True` at the top of the file to visualise a Bird's Eye View of the first frame in each split.

---

### Step 2 — Generate BEV images

```bash
python generate_bev.py
```

Converts each frame's registered LiDAR point cloud into a 1000×1000 px top-down (Bird's Eye View) image. Points are height-coloured using the `plasma` colormap. Annotated bounding boxes are overlaid (solid = visible, dashed = occluded). The `bev_path` for each frame is written back into the metrics JSON.

- **Input:** `data/metrics/{split}_metrics.json` + PCD files
- **Output:** `data/{split}/bev/*_bev.png`; `bev_path` field added to metrics JSON
- **Config:** `X_RANGE`, `Y_RANGE`, `RESOLUTION`, `Z_MIN/Z_MAX` at the top of the file.

> Only required if you intend to produce the BEV dataset variant (`INCLUDE_LIDAR = False` in Step 4).

---

### Step 3 — Generate QA pairs

```bash
python generate_qa.py
```

Sends each frame's structured metrics to GPT-4o via the OpenAI Batch API and generates exactly 10 QA triplets per frame — 4 maneuver, 3 counting, 3 spatial — across four distinct personas (Autonomous System Log, Driving Instructor, Co-Pilot, Physics Engine).

- **Input:** `data/metrics/{train,val,test}_metrics.json`
- **Output:** `data/raw/batch_input.jsonl` (submitted to OpenAI); the batch job ID is printed and should be saved to `.env` as `BATCH_JOB_ID`
- **Config:**
  - `BATCH_TEST = True` — process only 1 frame (recommended for testing, avoids cost)
  - `REALTIME_TEST = True` — use the synchronous API instead of batch

---

### Step 4 — Validate and format

```bash
python validate_qa.py
```

Polls the OpenAI Batch API until the job completes, downloads results, cross-references every generated object against ground truth to filter hallucinations, then formats surviving samples into Qwen-VL SFT conversation JSON.

- **Input:** `data/raw/batch_results.jsonl` + `data/metrics/{split}_metrics.json`
- **Output:** `data/datasets/d2_v2x_{split}[_bev][_ego][_nocot].json`
- **Config (dataset variants):**

| Flag | Default | Effect |
|------|---------|--------|
| `INCLUDE_LIDAR` | `True` | `False` → swap LiDAR for BEV image (requires Step 2) |
| `V2X` | `True` | `False` → ego-vehicle camera only, no infrastructure sensors |
| `COT` | `True` | `False` → omit `<think>` chain-of-thought blocks |

---

## Repository Structure

```
D2-V2X/
├── utils/
│   ├── requirements.txt      # Python dependencies for data pipeline
│   ├── parse_data.py         # Step 1 – parse raw annotations into metrics JSON
│   ├── generate_bev.py       # Step 2 – LiDAR point clouds → BEV PNG images
│   ├── generate_qa.py        # Step 3 – GPT-4o QA generation via Batch API
│   ├── validate_qa.py        # Step 4 – validation + Qwen-VL SFT formatting
│   ├── feature_extractor.py  # CenterPoint model wrapper (LiDAR feature extraction)
│   └── centerpoint.py        # CLI: extract CenterPoint features from dataset JSONs
├── qwen/                     # Qwen3-VL-8B model config and tokenizer
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   └── ...                   # *.safetensors weights are git-ignored
├── Datasets/
│   ├── Main Dataset/         # d2_v2x_{train,val,test}.json       (V2X + LiDAR + CoT)
│   ├── Direct Inference Dataset/  # *_nocot.json                  (V2X + LiDAR, no CoT)
│   └── Single-View Dataset/  # *_ego.json                        (vehicle camera only)
└── data/                     # raw sensor data – git-ignored
```

---

## Dataset Format

Each sample in the output JSON follows the Qwen-VL SFT conversational format:

```json
{
  "id": "16886266631410084_1",
  "file_metadata": {
    "image_paths": ["./data/train/images/...jpg", "..."],
    "lidar_path":  "./data/train/point_clouds/...pcd"
  },
  "conversations": [
    {
      "from": "user",
      "value": "Image 1: <image>\nImage 2: <image>\n...\nLiDAR: <lidar>\nIs it safe to turn left?"
    },
    {
      "from": "assistant",
      "value": "<think>\nInfrastructure sensors detect a van occluded by a truck...\n</think>\nNegative. It is unsafe to proceed.\n```json\n{\"task_type\": \"maneuver\", \"decision\": \"yield\", ...}\n```"
    }
  ]
}
```

Each assistant response contains:
- `<think>` block — step-by-step reasoning (CoT variant only)
- Natural language answer in the model's persona voice
- Structured `metrics` JSON with `task_type`, `decision`, `hazard_level`, `count`, and `grounded_objects`

---

## LiDAR Feature Extraction (CenterPoint)

`centerpoint.py` and `feature_extractor.py` support the LiDAR Adapter component. They run separately from the dataset generation pipeline, after `validate_qa.py` has produced the final dataset JSONs.

CenterPoint processes each sample's registered PCD file and extracts intermediate neural features (voxel reader → 3D sparse backbone → RPN neck), saving them as `.safetensors` files for use by the LiDAR Adapter during fine-tuning.

```bash
# Requires: CenterPoint repo cloned to utils/CenterPoint/ + pretrained weights
python utils/centerpoint.py \
    --mode train \
    --json_name d2_v2x_train.json \
    --dataset_dir "./Datasets/Main Dataset" \
    --data_root ./ \
    --checkpoint utils/CenterPoint/epoch_20.pth \
    --config utils/CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py
```

Features are saved to `data/tumtraf_features/{mode}/{sample_id}.safetensors`.

---

## Sensors

| Sensor | Source | Role |
|--------|--------|------|
| Basler south1 8mm | Infrastructure | Primary intersection view |
| Basler south2 8mm | Infrastructure | Secondary south view |
| Basler north 8mm | Infrastructure | Opposing approach |
| Basler east 8mm | Infrastructure | Cross-traffic view |
| Basler 16mm | Vehicle (ego) | Driver perspective |
| Ouster LiDAR | Infrastructure | 3-D scene reconstruction |
| RoboSense LiDAR | Vehicle (ego) | Ego-vehicle surroundings |

The two LiDAR streams are pre-fused into a single registered point cloud (`s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered`).
