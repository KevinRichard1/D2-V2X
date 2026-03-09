'''Generate QA Pairs from parsed data'''
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

TEST = True  # Set to True for real-time API
LIMIT = 10 if TEST else 800
INPUT_DIR = "../data/processed"
OUTPUT_FILE = "../data/raw_dataset.json"
MODEL = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an expert AI data engineer for V2X (Vehicle-to-Everything) Cooperative Perception.
Your task: Convert JSON multi-sensor frames into exactly 10 high-quality QRA triplets for Qwen3-VL fine-tuning.

OUTPUT FORMAT (STRICT):
You MUST output a valid JSON array `[` ... `]`. Do not output loose JSON objects.

MODALITY TOKEN RULES (MANDATORY):
- Image data (bbox, visibility status, camera name): wrap in <|vision_start|>...<|vision_end|>.
- Spatial/LiDAR data (distance_m, x, y, point cloud): wrap in <|lidar_start|>...<|lidar_end|>.
- Multi-modal fusion: Use BOTH tags appropriately in the same answer.
- NO TAGS IN THINKING: NEVER use <|vision_start|> or <|lidar_start|> tags inside the <think> block. Save them entirely for the final Answer string.

THINKING PROTOCOL (SPATIAL LOGIC):
Inside the <think> tag, you must execute this 3-step logic:
1. SENSOR IDENTIFICATION: Explicitly name the sensor (e.g., 'vehicle_camera_basler_16mm' vs 's110_camera_basler_south2_8mm'). Do not just say "the vehicle".
2. METADATA EXTRACTION: Extract the exact object ID, metric coordinates (x, y), and bbox arrays. If bboxes are empty, you MUST still extract the (x, y) coordinates.
3. SPATIAL CROSS-CHECK & EXACT STRING MATCHING: 
   - CRITICAL PERSPECTIVE RULE: The `visibility` field (e.g., "occluded_by_...") applies ONLY to the ego-vehicle's perspective. 
   - INFRASTRUCTURE VANTAGE (OCCLUSION): If an object is occluded from the ego-vehicle but visible to infrastructure, DO NOT paraphrase. You MUST use this EXACT string template: "The ego-vehicle cannot see [Object_ID] because it is physically blocked by [Occluder_ID], BUT the infrastructure sensor [Sensor_Name] has a clear view and detects it at [bbox]."
   - STRICT FOV RULE: If `visibility` is "clear" but there is no vehicle bounding box, DO NOT paraphrase. You MUST use this EXACT string template: "The object is outside the ego-vehicle's Field of View (FOV), but the infrastructure sensor captures it clearly at [bbox]." (Note: If the object is truncated, append the exact sentence "Note: This bounding box is truncated." at the end. Do NOT alter the main template.)
   - COMPLETE BLIND SPOT RULE: If the `bboxes` object is entirely empty `{}` for all sensors, DO NOT put coordinates in the vision tag. Use this exact structure: "<|vision_start|>The object is currently outside the visual field of both the ego-vehicle and infrastructure cameras.<|vision_end|> <|lidar_start|>Its spatial position is tracked via LiDAR at x: [x], y: [y] at a distance of [distance_m]m.<|lidar_end|>"
   - Do not hallucinate physics. Truncation means the object is at the edge of the camera frame, not occluded by distance.

CORE OBJECTIVES:
- Emphasize the exact value of V2X: Explicitly contrast the ego-vehicle's blind spots (occlusions/FOV limits) against the infrastructure's clear vantage point (presence of s110 bboxes).
- NEVER invent or truncate IDs. Use 'car_7e29ba6e' exactly as written.

QUESTION VARIETY RULE: Do not repeat the same question structure. Vary the instruction and input fields based on the context:

- If occluded: Ask about "occlusion resolution" or "safety risk."
- If outside FOV: Ask about "blind spot coverage" or "infrastructure assistance."
- If visible to both: Ask about "cooperative perception validation" or "redundancy."

STRICT ADHERENCE: If you provide any text outside of the JSON array, your response is considered a failure. Do not include markdown code block syntax (```json). Return ONLY the raw JSON array.

EXAMPLE OUTPUT:
[
  {
    "instruction": "Evaluate the visibility of car_4eab2f36 from the vehicle vs infrastructure.",
    "input": "Can the vehicle see car_4eab2f36, or is infrastructure data required?",
    "output": "<think>The vehicle_camera_basler_16mm has no bbox for car_4eab2f36 because its visibility field states it is 'occluded_by_car_7e29ba6e'. However, it exists in 3D space at (x: 42.12, y: 7.17). The infrastructure sensor s110_camera_basler_south2_8mm has a clear vantage point and captures it at bbox [914, 297, 973, 383].</think>Answer: <|vision_start|>The ego-vehicle cannot see car_4eab2f36 because it is physically blocked by car_7e29ba6e, BUT the infrastructure sensor s110_camera_basler_south2_8mm has a clear view and detects it at [914, 297, 973, 383].<|vision_end|> <|lidar_start|>The object's true spatial position is x: 42.12, y: 7.17 at a distance of 42.72m.<|lidar_end|>"
  }
]
"""

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "triplets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "rationale": {"type": "string"},
                    "answer": {"type": "string"}
                },
                "required": ["prompt", "rationale", "answer"]
            }
        }
    }
}

def simplify_frame(frame):
    '''Strip unnecessary data to save tokens'''
    return {
        "frame_id": frame["frame_id"],
        "objects": [
            {
                "id": obj.get("id", "unknown"),
                "x": obj.get("x", 0.0),
                "y": obj.get("y", 0.0),
                "type": obj.get("type", "unknown"),
                "size": obj.get("size", "unknown"),
                "distance_category": obj.get("distance_category", "unknown"),
                "distance_m": obj.get("distance_m", 0.0),
                "position": obj.get("position", "unknown"),
                "visibility": obj.get("visibility", "unknown"),
                "length_m": obj.get("length_m", 0.0),
                "width_m": obj.get("width_m", 0.0),
                "height_m": obj.get("height_m", 0.0),
                "heading": obj.get("heading", "unknown"),
                "detected_by": obj.get("detected_by", "unknown"),
                "primary_view": obj.get("primary_view", "unknown"),
                "density": obj.get("density", "unknown"),
                "bboxes": obj.get("bboxes", {}),
                "is_truncated": obj.get("is_truncated", {})
            }
            for obj in frame.get("objects", [])
        ]
    }

def run_realtime(frames):
    '''Real-time API processing for testing'''
    dataset = []
    backup_file = "realtime_backup.jsonl"

    # Remove existing backup file if it exists
    if os.path.exists(backup_file):
        os.remove(backup_file)
    
    for frame in frames:
        print(f"Processing Frame {frame['frame_id']}...")
        try:
            response = client.responses.create(
                model=MODEL,
                temperature=0.0,
                top_p=0.25,
                max_output_tokens=4096,
                instructions=SYSTEM_PROMPT,
                input=json.dumps(simplify_frame(frame)),                
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "v2x_qa",
                        "strict": True,
                        "schema": QA_SCHEMA
                    }
                }
            )

            # Extract validated JSON from response
            raw_content = response.output_text
            parsed_json = json.loads(raw_content)
            triplets = parsed_json.get("triplets", [])
            
            dataset.extend(triplets)
            
            # Checkpoint to disk
            with open(backup_file, "a") as f:
                for t in triplets:
                    f.write(json.dumps(t) + "\n")
                    
        except Exception as e:
            print(f"Failed to process frame {frame.get('frame_id')}: {e}")
            
    return dataset

def run_batch(frames):
    '''Batch API processing for full dataset'''
    batch_file = "batch_input.jsonl"
    with open(batch_file, "w") as f:
        for frame in frames:
            req = {
                "custom_id": f"frame_{frame['frame_id']}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "temperature": 0.0,
                    "top_p": 0.25,
                    "max_output_tokens": 4096,
                    "instructions": SYSTEM_PROMPT,
                    "input": json.dumps(simplify_frame(frame)),
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "v2x_qa",
                            "strict": True,
                            "schema": QA_SCHEMA
                        }
                    }
                }
            }
            f.write(json.dumps(req) + "\n")
    
    # Upload and Trigger
    file_batch = client.files.create(file=open(batch_file, "rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=file_batch.id,
        endpoint="/v1/responses",
        completion_window="24h"
    )
    print(f"Batch Job Created. ID: {batch_job.id}")

if __name__ == "__main__":
    if TEST:
        # Test on subset of specific file
        test_path = os.path.join(INPUT_DIR, "train_metrics.json")
        if not os.path.exists(test_path):
            print(f"Error: {test_path} not found.")
        else:
            with open(test_path, "r") as f:
                test_frames = json.load(f)[:LIMIT]

            results = run_realtime(test_frames)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Processed {len(results)} QA pairs.")
    else:
        # Consolidate all files
        all_frames = []
        for file_name in os.listdir(INPUT_DIR):
            if file_name.endswith(".json"):
                file_path = os.path.join(INPUT_DIR, file_name)

                with open(file_path, "r") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    all_frames.extend(data)
                    print(f"Loaded {len(data)} frames from {file_name}")
                else:
                    print(f"Warning: {file_name} is not a list.")

        if all_frames:
            print(f"Total frames collected: {len(all_frames)}. Submitting batch job")
            run_batch(all_frames)
        else:
            print("No valid JSON files found in directory.")