import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoProcessor
from utils.metrics import compute_metrics

def run_inference(model, processor, dataset, output_file):
    model.eval()
    results = []

    for idx in tqdm(range(len(dataset)), desc="Generating Predictions"):
        messages, lidar_tensors, sample_id = dataset[idx]
        
        if lidar_tensors is not None:
            num_lidar_tokens = 1024
            replacement = "<lidar>" * num_lidar_tokens

            for message in messages:
                if isinstance(message.get("content"), list):
                    for block in message["content"]:
                        if block.get("type") == "text" and "<lidar>" in block.get("text", ""):
                            block["text"] = block["text"].replace("<lidar>", replacement)
                elif isinstance(message.get("content"), str):
                    if "<lidar>" in message["content"]:
                        message["content"] = message["content"].replace("<lidar>", replacement)

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        images = []
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, list):
                images.extend([item['image'] for item in content if item.get('type') == 'image'])

        inputs = processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(model.model.device)

        if lidar_tensors is not None:
            if lidar_tensors.ndim == 4:
                inputs["lidar_features"] = lidar_tensors.to(model.model.device)
            else:
                inputs["lidar_features"] = lidar_tensors.unsqueeze(0).to(model.model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1
            )

        # Decode output
        new_tokens = generated_ids[0]
        pred_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Store
        raw_sample = dataset.samples[idx]
        gt_text = raw_sample['conversations'][1]['value']

        results.append({
            "id": sample_id,
            "prediction": pred_text.strip(),
            "ground_truth": gt_text
        })

        if idx > 0 and idx % 50 == 0:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file.replace(".json", "_partial.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)

    # Save to output_file
    print(f"Saving {len(results)} predictions to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

def run_evaluation(prediction_file, tokenizer, gt_file_path=None):
    with open(prediction_file, 'r') as f:
        data = json.load(f)

    raw_preds = []
    raw_gts = []
    
    # Extract strings from JSON
    if isinstance(data, dict):
        if not gt_file_path:
            raise ValueError("Original dataset JSON pathis required to evaluate zero-shot predictions.")
        
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            gt_dataset = json.load(f)
        
        # Create lookup map for ground truths by ID
        gt_map = {}
        for item in gt_dataset:
            sample_id = str(item.get('id', ''))
            conversations = item.get('conversations', [])
            if len(conversations) > 1:
                gt_map[sample_id] = conversations[1].get('value', '')

        # Pair predictions with ground truths
        for sample_id, pred_text in data.items():
            raw_preds.append(pred_text)
            raw_gts.append(gt_map.get(str(sample_id), ""))
    elif isinstance(data, list):
        raw_preds = [item['prediction'] for item in data]
        raw_gts = [item['ground_truth'] for item in data]

    # Convert to token IDs
    encoded_preds = tokenizer(raw_preds, padding=True, return_tensors="np").input_ids
    encoded_gts = tokenizer(raw_gts, padding=True, return_tensors="np").input_ids

    eval_pred = (encoded_preds, encoded_gts)

    results = compute_metrics(eval_pred, tokenizer)
    
    print("--- Evaluation Results ---")
    for key, value in sorted(results.items()):
        display_name = key.replace("eval_", "").replace("_", " ").title()
        print(f"{display_name:<30}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", type=str, default='./qwen')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/stage2/final_model/lora')
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--mode", type=str, default="d2v2x") # bev, ego, nocot

    parser.add_argument("--json_path", type=str, default="./data/datasets/d2_v2x_test.json")
    parser.add_argument("--img_path", type=str, default="./data/val/images")
    parser.add_argument("--test_feature_path", type=str, default="./data/tumtraf_features")
    parser.add_argument("--inference_save_path", default="results.json")

    args = parser.parse_args()

    print(f"Loading processor from {args.qwen_path}...")
    processor = AutoProcessor.from_pretrained(args.qwen_path)
    processor.tokenizer.padding_side = "left"

    output_file = args.inference_save_path

    if args.inference:
        # Load dataset and model
        from models.d2v2x_model import D2V2XModel
        from data_pipeline.dataset import D2V2XDataset

        print(f"Loading Dataset in {args.mode} mode...")

        dataset = D2V2XDataset(
            json_path=args.json_path,
            data_root=args.img_path,
            feature_dir=args.test_feature_path,
            mode=args.mode,
            is_training=False
        )

        print(f"Loading model from {args.checkpoint_path}...")
        model = D2V2XModel(
            base_model_path=args.qwen_path,
            adapter_path=args.checkpoint_path,
            mode=args.mode)
        model.to("cuda")

        # Run inference
        run_inference(model, processor, dataset, output_file)
    
    if args.eval:
        # Load tokenizer
        tokenizer = processor.tokenizer
        run_evaluation(output_file, tokenizer, gt_file_path=args.json_path)