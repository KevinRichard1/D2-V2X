'''Test the model using image-only zero-shot inference'''
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils.dataset import D2V2XDataset
from tqdm import tqdm
import argparse

def custom_collate_fn(batch):
    messages = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    sample_ids = [item[2] for item in batch]
    return messages, labels, sample_ids

def run_zero_shot(args):
    # Load Model and Processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"

    # Setup Data
    dataset = D2V2XDataset(
        args.json_path,
        args.data_root,
        args.feature_dir,
        processor,
        mode=["image_only", "zero_shot"],
        is_training=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=custom_collate_fn
    )

    model.eval()
    results = {}

    for batch_idx, (messages_batch, _, sample_id_batch) in enumerate(tqdm(dataloader)):
        texts = []
        batch_images = []

        # Process each sample in batch
        for messages in messages_batch:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)

            user_content = messages[0]['content']
            sample_images = [item['image'] for item in user_content if item['type'] == 'image']
            batch_images.extend(sample_images)

        
        inputs = processor(
            text=texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=4096)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Map responses back to ID
        for sid, response in zip(sample_id_batch, responses):
            results[sid] = response
            print(f"ID: {sid} | Response: {response[:80]}...")

        if batch_idx % 10 == 0:
            os.makedirs(args.output_path, exist_ok=True)
            with open(os.path.join(args.output_path, "zero_shot_results_partial.json"), "w") as f:
                json.dump(results, f, indent=4)

    # Save all results
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