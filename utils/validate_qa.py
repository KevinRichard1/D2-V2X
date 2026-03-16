'''Validate generated QA pairs against original frame data'''
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
BATCH_JOB_ID = os.getenv("BATCH_JOB_ID")

METRICS_DIR = "../data/metrics"
BATCH_OUTPUT_FILE = "../data/raw/batch_results.jsonl"
FINAL_DATASET_DIR = "../data/datasets"

# Dataset variations
INCLUDE_LIDAR = True        # Set false for Image + BEV only
V2X = True                  # Set false for ego-vehicle only
COT = True                  # Set false for no chain-of-thought

def wait_for_batch_completion(batch_id, poll_interval_sec=60):
    '''Polls the OpenAI API until the batch job completes or fails.'''
    print(f"Polling Batch Job: {batch_id}...")
    terminal_states = ["failed", "expired", "cancelled"]

    # Polling loop
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed": 
            print("Batch job completed successfully.")
            return batch.output_file_id
        elif batch.status in terminal_states:
            raise RuntimeError(f"Batch job terminated with status: {batch.status}")
        else:
            print(f"Current status: {batch.status}. Checking again in {poll_interval_sec} seconds...")
            time.sleep(poll_interval_sec)

def download_batch_results(file_id, output_path):
    '''Downloads the output .jsonl file and saves it .'''
    print(f"Downloading results...")
    file_response = client.files.content(file_id)
    file_content = file_response.text

    with open(output_path, "w") as f:
        f.write(file_content)
    print(f"Results saved to {output_path}")

def extract_responses(jsonl_path):
    '''Reads the downloaded JSONL file and extracts the validated JSON string.'''
    print("Extracting generated responses from JSONL...")
    extracted_data = {}
    with open(jsonl_path, "r") as f:
        # Parse each line as JSON
        for line in f:
            record = json.loads(line)
            frame_id = record.get("custom_id")

            # Extract response string
            text = ""
            body = record.get("response", {}).get("body", {})
            for item in body.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            text = c.get("text", "")
                            break

            # Save with frame_id mapped to triplet
            try:
                extracted_data[frame_id] = json.loads(text)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for frame {frame_id}: {e}")
    print(f"Loaded {len(extracted_data)} responses from batch output.")
    return extracted_data

# Helper for dynamic tolerance based on GT tolerance
def get_dynamic_tolerance(distance_m):
    if distance_m < 15.0:
        return 0.5
    elif distance_m < 40.0:
        return 1.5
    else:
        return 3.0

def validate_and_merge(extracted_data, ground_truth_path):
    '''Cross-references generated outputs with original metrics to ensure data integrity
    and merges image/lidar paths with the QA pairs.'''
    print("Validating generated QA against ground truth...")
    validated_samples = []
    discarded_count = 0
    
    with open(ground_truth_path, "r") as f:
        gt_metrics = json.load(f)

    # Iterate through each frame in ground truth metrics
    for frame in gt_metrics:
        base_id = str(frame.get("frame_id"))
        
        possible_keys = [base_id, f"frame_{base_id}", base_id.replace("frame_", "")]
        matched_key = next((k for k in possible_keys if k in extracted_data), None)

        if matched_key:
            raw_data = extracted_data[matched_key]
            triplets = raw_data.get("triplets", []) if isinstance(raw_data, dict) else raw_data
            for triplet in triplets:
                # Compare generated metrics with gt_metrics
                struct_metrics = triplet.get("structured_metrics", {})
                gen_objects = struct_metrics.get("grounded_objects", [])
                gen_count = struct_metrics.get("count", 0)
                gt_objects = frame.get("objects", [])

                is_valid = True

                # Check internal consistency
                if gen_count != len(gen_objects):
                    is_valid = False
                
                # Check for hallucination
                if is_valid and gen_count > 0:
                    available_gt_objects = gt_objects.copy()

                    for gen_obj in gen_objects:
                        match_found = False
                        for i, gt_obj in enumerate(available_gt_objects):
                            # Check if type matches and distance is within tolerance
                            type_match = gt_obj.get("type") == gen_obj.get("type")
                            dist_diff = abs(gt_obj.get("distance_m", 0) - gen_obj.get("distance_m", 0))
                            dist_tolerance = get_dynamic_tolerance(gt_obj.get("distance_m", 0))
                            
                            if type_match and dist_diff <= dist_tolerance:
                                match_found = True
                                available_gt_objects.pop(i) 
                                break
                        
                        if not match_found:
                            is_valid = False
                            break
                            
                # Skip appending if validation failed
                if not is_valid:
                    discarded_count += 1
                    continue

                merged_sample = {
                    "frame_id": base_id,
                    "images": frame.get("images"),
                    "lidar_files": frame.get("lidar_files"),
                    "bev_path": frame.get("bev_path"),
                    "prompt": triplet.get("prompt"),
                    "rationale": triplet.get("rationale"),
                    "answer": triplet.get('answer'),
                    "structured_metrics": struct_metrics
                }
                validated_samples.append(merged_sample)
        else:
            pass

    print(f"Validation complete: Kept {len(validated_samples)} | Discarded {discarded_count} hallucinations.")
    return validated_samples

def format_for_qwen(merged_data, include_lidar=True, v2x=True, cot=True):
    '''Converts the merged data into the specific conversational JSON format 
    required by Qwen-VL for fine-tuning.'''
    print("Formatting dataset for Qwen-VL...")
    dataset = []
    frame_counts = {}

    for sample in merged_data:
        frame_id = sample.get("frame_id")
        
        frame_counts[frame_id] = frame_counts.get(frame_id, 0) + 1
        unique_id = f"{frame_id}_{frame_counts[frame_id]}"

        prompt_parts = []
        file_metadata = {"image_paths": []}
        rationale = sample.get('rationale', '')
        answer = sample.get('answer', '')
        metrics = sample.get('structured_metrics', {})
        formatted_metrics = json.dumps(metrics, indent=2)

        # Add all images
        for i, path in enumerate(sample.get("images", {}).values()):
            prompt_parts.append(f"Image {i+1}: <image>") 
            file_metadata["image_paths"].append(path)

        if not include_lidar:
            # Add BEV image (generated by generate_bev.py)
            bev_path = sample.get("bev_path")
            if bev_path:
                prompt_parts.append("BEV: <image>")
                file_metadata["image_paths"].append(bev_path)
            else:
                print(f"[WARN] No bev_path for frame {sample.get('frame_id')}. "
                      "Run generate_bev.py before validate_qa.py.")
        elif not v2x:
            # Include only vehicle image and lidar
            prompt_parts = [prompt_parts[-1]]
            file_metadata["image_paths"] = [file_metadata["image_paths"][-1]]
            prompt_parts.append("LiDAR: <lidar>")
            file_metadata["lidar_path"] = sample.get("lidar_files")[0]
        else:
            prompt_parts.append("LiDAR: <lidar>")
            file_metadata["lidar_path"] = sample.get("lidar_files")[2]

        # Assemble prompt with original question
        prompt_parts.append(sample.get("prompt"))       
            
        if cot:
            combined_response = (
                f"<think>\n{rationale}\n</think>\n"
                f"{answer}\n"
                f"```json\n{formatted_metrics}\n```"
            )
        else:
            combined_response = (
                f"{answer}\n"
                f"```json\n{formatted_metrics}\n```"
            )

        conversation_sample = {
            "id": unique_id,
            "file_metadata": file_metadata,
            "conversations": [
                {
                    "from": "user",
                    "value": "\n".join(prompt_parts)
                },
                {
                    "from": "assistant",
                    "value": combined_response
                }
            ]
        }

        dataset.append(conversation_sample)   
    
    return dataset

def main():
    if not BATCH_JOB_ID:
        raise ValueError("BATCH_JOB_ID not found in environment variables.")

    # Wait and Download
    output_file_id = wait_for_batch_completion(BATCH_JOB_ID)
    download_batch_results(output_file_id, BATCH_OUTPUT_FILE)
    
    # Extract and Validate for each split
    extracted_data = extract_responses(BATCH_OUTPUT_FILE)
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\nProcessing Split: {split.upper()}")
        
        ground_truth_path = os.path.join(METRICS_DIR, f"{split}_metrics.json")
        if not os.path.exists(ground_truth_path):
            print(f"Warning: {ground_truth_path} not found. Skipping {split}.")
            continue

        # Load ground truth for specific splits
        with open(ground_truth_path, "r") as f:
            gt_frames = json.load(f)
        split_ids = {str(f.get("frame_id")) for f in gt_frames}

        split_specific_extracted = {
            k: v for k, v in extracted_data.items() 
            if k in split_ids or k.replace("frame_", "") in split_ids
        }

        # Use validation to filter
        validated_split_data = validate_and_merge(split_specific_extracted, ground_truth_path)

        # Format for Qwen-VL SFT
        dataset_split = format_for_qwen(validated_split_data, include_lidar=INCLUDE_LIDAR, v2x=V2X, cot=COT )

        lidar_tag = "" if INCLUDE_LIDAR else "_bev"
        v2x_tag = "" if V2X else "_ego"
        cot_tag = "" if COT else "_nocot"

        # Save the final JSON
        output_path = os.path.join(FINAL_DATASET_DIR, f"d2_v2x_{split}{lidar_tag}{v2x_tag}{cot_tag}.json")
        with open(output_path, "w") as f:
            json.dump(dataset_split, f, indent=4)
        print(f"Successfully saved {split} dataset to {output_path} with {len(dataset_split)} samples.")

if __name__ == "__main__":
    main()