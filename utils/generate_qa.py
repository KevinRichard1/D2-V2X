'''Generate QA Pairs from parsed data'''
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

TEST = True  # Set to True for real-time API
LIMIT = 1 if TEST else 800
INPUT_DIR = "../data/processed"
OUTPUT_FILE = "../data/raw_dataset.json"
MODEL = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an expert AI data engineer for V2X (Vehicle-to-Everything) Cooperative Perception.
Your task: Convert JSON multi-sensor frames into exactly 10 high-quality QRA triplets for Qwen-VL fine-tuning.

OUTPUT FORMAT (STRICT):
You MUST output a valid JSON array `[` ... `]`. Do not output loose JSON objects. Each object must have exactly three keys: "prompt", "rationale", and "answer".

MODALITY TOKEN RULES (MANDATORY):
- Visual Grounding: DO NOT wrap text in vision or lidar tags. Output bounding boxes as raw arrays [xmin, ymin, xmax, ymax].

THINKING PROTOCOL (SPATIAL LOGIC):
Inside the "rationale" field, write out a natural, step-by-step internal monologue:
- "First, I need to check the ego-vehicle camera to see if [ID] is visible..." (State visibility and coords).
- "Since it is occluded by [Occluder], I need to check the infrastructure sensors..."
- "The s110 camera clearly captures it at [bbox], so I will structure my final answer to highlight this V2X redundancy."
1. SENSOR IDENTIFICATION: Explicitly name the sensor (e.g., 'vehicle_camera_basler_16mm' vs 's110_camera_basler_south2_8mm'). Do not just say "the vehicle".
2. METADATA EXTRACTION: Extract the exact object ID, metric coordinates (x, y), and bbox arrays. If bboxes are empty, you MUST still extract the (x, y) coordinates.
3. SPATIAL CROSS-CHECK & EXACT STRING MATCHING FOR THE ANSWER: 
   - RULE 1: COMPLETE BLIND SPOT (CHECK THIS FIRST): If the `bboxes` object is entirely empty `{}` or missing for BOTH the ego-vehicle and infrastructure sensors, DO NOT use any other rule. Use this exact structure: "The object is currently outside the visual field of both the ego-vehicle and infrastructure cameras. Its spatial position is tracked via LiDAR at x: [x], y: [y] at a distance of [distance_m]m."
   - RULE 2: INFRASTRUCTURE VANTAGE (OCCLUSION): If an object is occluded from the ego-vehicle but has a valid bbox from the infrastructure sensor, use this EXACT string template: "The ego-vehicle cannot see [Object_ID] because it is physically blocked by [Occluder_ID], BUT the infrastructure sensor [Sensor_Name] has a clear view and detects it at [insert actual bbox coordinates here]." (Note: If truncated, append "Note: This bounding box is truncated." at the end.)
   - RULE 3: STRICT FOV: If `visibility` is "clear" but there is no vehicle bounding box, AND the infrastructure sensor has a valid bbox, use this EXACT string template: "The object is outside the ego-vehicle's Field of View (FOV), but the infrastructure sensor captures it clearly at [insert actual bbox coordinates here]." (Note: If truncated, append "Note: This bounding box is truncated." at the end.)
   - CRITICAL: Never literally output the string "[bbox]". Always replace it with the actual coordinate array (e.g., [100, 200, 300, 400]). If you do not have coordinate numbers, you must fall back to RULE 1.

CORE OBJECTIVES:
- Emphasize the exact value of V2X: Explicitly contrast the ego-vehicle's blind spots (occlusions/FOV limits) against the infrastructure's clear vantage point.
- NEVER invent or truncate IDs. Use 'car_7e29ba6e' exactly as written.

PROMPT DIVERSITY ENGINE (CRITICAL - READ CAREFULLY):
You MUST aggressively randomize the syntactic structure of the "prompt" field. Do NOT repeat question formats. 
Mix and match these styles:
- Direct spatial queries: "Where is [ID] located?" or "Give me the coordinates for [ID]."
- Conversational safety queries: "Is there anything hiding behind the truck that I should know about?" or "Can the ego vehicle safely see [ID]?"
- Sensor-specific commands: "Cross-reference the infrastructure camera with the ego-vehicle for [ID]." or "What does the s110 camera see regarding [ID]?"

FORBIDDEN PHRASES: You are strictly FORBIDDEN from starting prompts with:
- "Analyze the occlusion risk..."
- "How does infrastructure assist..."
- "Evaluate the visibility of..."
- "Discuss the detection of..."

STRICT ADHERENCE: If you provide any text outside of the JSON array, your response is considered a failure. Do not include markdown code block syntax (```json). Return ONLY the raw JSON array.

EXAMPLE OUTPUT:
[
  {
    "prompt": "Is there anything hiding behind car_7e29ba6e that the ego vehicle should be aware of regarding car_4eab2f36?",
    "rationale": "First, I need to check the ego-vehicle camera to see if car_4eab2f36 is visible. I see that vehicle_camera_basler_16mm has no bounding box for it because it is marked as 'occluded_by_car_7e29ba6e'. However, checking the spatial data, the object does exist in 3D space at (x: 42.12, y: 7.17). Since it is occluded from the ego-vehicle, I need to check the infrastructure sensors to see if they can resolve this blind spot. Looking at the s110_camera_basler_south2_8mm data, it clearly captures the car at bbox [914, 297, 973, 383]. I will structure my final answer to highlight this V2X redundancy.",
    "answer": "The ego-vehicle cannot see car_4eab2f36 because it is physically blocked by car_7e29ba6e, BUT the infrastructure sensor s110_camera_basler_south2_8mm has a clear view and detects it at [914, 297, 973, 383]. The object's true spatial position is x: 42.12, y: 7.17 at a distance of 42.72m."
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
                "required": ["prompt", "rationale", "answer"],
                "additionalProperties": False
            }
        }
    },
    "required": ["triplets"],
    "additionalProperties": False
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
                temperature=0.3,
                top_p=0.7,
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