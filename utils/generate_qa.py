'''Generate QA Pairs from parsed data'''
import os
import json
from openai import OpenAI
from dotenv import load_dotenv, set_key

load_dotenv(override=True)

REALTIME_TEST = False  # Set to True for real-time API
BATCH_TEST = True # Set to true to use batch API for a single frame
INPUT_DIR = "../data/metrics"
OUTPUT_FILE = "../data/raw_dataset.json"
MODEL = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an expert AI data engineer for V2X (Vehicle-to-Everything) Cooperative Perception.
Your task: Convert JSON multi-sensor frames into exactly 10 high-quality QRA (Question, Rationale, Answer) samples for Qwen-VL fine-tuning.

OUTPUT FORMAT (STRICT):
You MUST output a valid JSON object containing a triplets array
{
  "prompt": "The natural language question.",
  "rationale": "The step-by-step reasoning process.",
  "answer": "The conversational answer provided to the driver/system.",
  "structured_metrics": {
    "task_type": "maneuver" | "counting" | "spatial",
    "decision": "safe | unsafe | yield | monitor",
    "hazard_level": "none | low | medium | high",
    "count": <integer>,
    "grounded_objects": [
      {
         "type": "type of object",
         "bbox": [xmin, ymin, xmax, ymax],
         "distance_m": <float>,
	     "sensor_id": "sensor source"
      }
    ]
  }
}

TASK DISTRIBUTION ENGINE (MANDATORY MIX):
To ensure global scene awareness, your 10 generated samples MUST follow this exact distribution of task types.
- 4x Driving Maneuver Tasks: Ask if a specific driving action is safe (e.g., turning left, proceeding through intersection). Base the danger on actual occluded objects in the JSON revealed by V2X/LiDAR.
- 3x Scene Counting/Filtering Tasks: Ask about quantities or classes in blind spots (e.g., "How many pedestrians are hidden from the ego-vehicle?").
- 3x Relational/Spatial Tasks: Ask about distances or relative positions of specific hazards.

THINKING PROTOCOL (SCENE AWARENESS LOGIC):
Inside the "rationale", write out a flowing, natural monologue:
1. Evaluate the *global scene* from the ego-vehicle's perspective (identify occluders and FOV limits).
2. Cross-reference infrastructure sensors and LiDAR data to expose hidden hazards.
3. Synthesize how these hidden objects impact the driving task.
- STRICT OCCLUDER CHECK & ID STRIPPING: Read the literal string in the "visibility" key. If it says "occluded_by_car_123", identify the occluder simply as "a car". You MUST strip all alphanumeric IDs from your rationale and answer (Write "hidden by a truck", NEVER write "hidden by truck_99a"). 
- VISIBILITY RULES: NEVER guess an occluder based on vehicle size. If "visibility" is "clear", treat it as a visible object, NOT a hidden hazard.
- STRICT SPATIAL GROUNDING: Include the exact (x, y) coordinates from the JSON in your rationale for every hidden hazard mentioned (e.g., 'Telemetry registers a medium car at x: -23.45, y: -11.1').
- REQUIRED SENSOR VERBS: Explicitly state why infrastructure was needed. You MUST use one of the following verbs to describe the sensor action: "broadcasts," "pings," "registers," "picks up," "telemetry shows," or "infrastructure confirms." (Example: 'Because the ego-camera only sees the side of the nearby car, infrastructure LiDAR pings the trailer at 16 meters...')
- STRICT ENVIRONMENTAL GROUNDING: Infer the ego-vehicle's scenario strictly from the "position" keys. Do not hallucinate environments (e.g., do not mention parking lots or highways unless explicitly supported by the position keys).

ANSWER GUIDELINES:
- Your "answer" should be fluid and conversational, explaining the situation clearly based on the rationale. 
- Keep rationales under 150 words while maintaining the spatial logic.
- Your "structured_metrics" MUST accurately reflect the answer. 
- If the task is a Counting task, the "count" field must be the total integer, and the "grounded_objects" array should list the specific objects counted.
- Visual Grounding: DO NOT wrap text in vision or lidar tags.
- Missing Bounding Boxes: If an object is detected by LiDAR but has no camera bounding box in the input JSON, output "bbox": []. DO NOT output [0,0,0,0].
- If a Maneuver task detects multiple hidden hazards, your "structured_metrics" must ground ALL of those hidden hazards and the "count" must reflect the total number of grounded objects, not just the closest one.
- Bounding Box Extraction: The bboxes field in the JSON is a dictionary. Extract the bounding box array corresponding to the camera listed in the object's primary_view. If bboxes is empty, if the camera key is missing, or if primary_view is "unknown", output "bbox": [].
- Empty Blind Spots: If a Maneuver task checks behind an occluder and finds NO hidden objects, set "decision": "Safe", "hazard_level": "None", "count": 0, and "grounded_objects": [].
- For counting tasks, ONLY count objects where the 'visibility' key contains an occluder. NEVER base a counting task on 'clear' objects
- Ensure all double quotes inside JSON strings are properly escaped with a backslash

PROMPT SYNTAX RANDOMIZATION & PERSONA MATRIX (CRITICAL):
To guarantee lexical diversity across the 10 samples, you MUST roleplay as one of these four personas for both the prompt and rationale. Embody the persona completely using first-person language ("I", "we", "my"). Speak directly to the driver or system. 

REQUIRED PERSONAS:
1. The Autonomous System Log: Clinical, robotic, data-driven. (e.g., "Diagnostic run. My V2X telemetry indicates a blocked vector...")
2. The Driving Instructor: Educational, cautious, instructional. (e.g., "Before we make that turn, notice how our line of sight is cut off...")
3. The Co-Pilot: Conversational, helpful, direct. (e.g., "I wouldn't switch lanes just yet. My infrastructure feed shows...")
4. The Physics Engine: Focuses purely on trajectories, coordinates, and spatial geometry. (e.g., "Calculating vectors. I am registering a blocked path...")

PERSONA GUARDRAILS:
- DO: Speak entirely in character.
- DO NOT: Ever name the persona or speak in the third person. 
- BAD EXAMPLE: "The co-pilot assessment indicates a blind spot..."
- GOOD EXAMPLE: "I am checking the blind spot..."

STRICT ADHERENCE: If you provide any text outside of the JSON array, your response is considered a failure. Do not include markdown code block syntax (```json). Return ONLY the raw JSON array.

EXAMPLE OUTPUT:

{
"triplets": [
    {
        "prompt": "Executing a self-diagnostic. Is the intended left turn path completely clear of hidden hazards?",
        "rationale": "System sensors indicate a severe blind spot. The ego-camera's line of sight is completely cut off because of visibility: occluded_by_truck_99a. Relying solely on onboard vision would be critical error. Accessing the V2X infrastructure LiDAR array reveals an oncoming hazard we cannot see: a medium car at coordinates x: 14.22, y: 31.05. Because this concealed vehicle is on a direct collision vector for the left turn, the maneuver must be aborted.",
        "answer": "Negative. It is unsafe to proceed. While the truck is blocking your camera's view, infrastructure LiDAR has detected a car hidden behind it at approximately 34 meters away. Yield immediately.",
        "structured_metrics": {
        "task_type": "maneuver",
        "decision": "yield",
        "hazard_level": "high",
        "count": 1,
        "grounded_objects": [
            {
            "type": "car",
            "bbox": [412, 290, 500, 380],
            "distance_m": 34.11,
            "sensor_id": "s110_lidar_ouster_south"
            }
        ]
        }
    }
]
}
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
                    "answer": {"type": "string"},
                    "structured_metrics": {
                        "type": "object",
                        "properties": {
                            "task_type": {"type": "string"},
                            "decision": {"type": "string"},
                            "hazard_level": {"type": "string"},
                            "count": {"type": "integer"},
                            "grounded_objects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "bbox": {
                                            "type": "array",
                                            "items": {"type": "number"}
                                        },
                                        "distance_m": {"type": "number"},
                                        "sensor_id": {"type": "string"}
                                    },
                                    "required": ["type", "bbox", "distance_m", "sensor_id"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["task_type", "decision", "hazard_level", "count", "grounded_objects"],
                        "additionalProperties": False
                    }
                },
                "required": ["prompt", "rationale", "answer", "structured_metrics"],
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
                "type": obj.get("type", "unknown"),
                "x": obj.get("x", 0.0),
                "y": obj.get("y", 0.0),
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
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=16384,
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
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_output_tokens": 16384,
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
    set_key(".env", "BATCH_JOB_ID", batch_job.id)
    print(f"Batch Job Created. ID: {batch_job.id}")

if __name__ == "__main__":
    if REALTIME_TEST:
        # Test on subset of specific file
        test_path = os.path.join(INPUT_DIR, "train_metrics.json")
        if not os.path.exists(test_path):
            print(f"Error: {test_path} not found.")
        else:
            with open(test_path, "r") as f:
                test_frames = json.load(f)[:1]

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
            if BATCH_TEST:
                frames_to_submit = all_frames[:1]
                print("Running batch test on a single frame...")
            else:
                frames_to_submit = all_frames
                print(f"Submitting batch job on {len(all_frames)} frames...")
            run_batch(frames_to_submit)
        else:
            print("No valid JSON files found in directory.")