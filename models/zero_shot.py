'''Test the model using image-only zero-shot inference'''
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils.dataset import D2V2XDataset
from tqdm import tqdm
import argparse

def run_zero_shot(args):
    # Load Model and Processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Setup Data
    dataset = D2V2XDataset(
        args.json_path,
        args.data_root,
        args.feature_dir,
        processor,
        mode=["image_only", "zero_shot"]
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    results = {}

    for messages, _, sample_id in tqdm(dataloader):
        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = [m[0]['content'][:-1] for m in messages]
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results[sample_id[0]] = response
        
        # Print first few to check logic
        print(f"ID: {sample_id[0]} | Response: {response[:100]}...")

    # Save results for comparison later
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "zero_shot_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test zero-shot model")
    parser.add_argument('--model_path', type=str, default='../qwen')
    parser.add_argument('--json_path', type=str, default='../data/datasets/d2_v2x_test.json')
    parser.add_argument('--data_root', type=str, default='../data/val/images')
    parser.add_argument('--feature_dir', type=str, default='../data/tumtraf_features')
    parser.add_argument('--output_path', type=str, default='../results')

    args = parser.parse_args()
    run_zero_shot(args)