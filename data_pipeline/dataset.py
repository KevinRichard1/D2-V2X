'''Dataset Loader'''
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file

class D2V2XDataset(Dataset):
    def __init__(self, json_path, data_root, feature_dir, mode, is_training):
        self.data_root = Path(data_root)
        self.feature_dir = Path(feature_dir)
        if isinstance(mode, str):
            self.mode = [mode]
        else:
            self.mode = mode
        self.is_training = is_training

        with open(json_path, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['id']
        user_query = sample['conversations'][0]['value']
        response = sample['conversations'][1]['value']
        img_paths = sample['file_metadata']['image_paths']

        # Load images
        images = []
        for p in img_paths:
            full_path = self.data_root / p.lstrip('./')
            images.append(Image.open(full_path).convert("RGB"))

        lidar_tensors = None

        if "image_only" in self.mode:
            # Clean up LiDAR files
            user_query = user_query.replace("LiDAR: <lidar>\n", "")
            user_query = user_query.replace("<lidar>", "[LiDAR data not available]")
        else:
            safetensor_path = self.feature_dir / f"{sample_id}.safetensors"
            safetensor_dict = load_file(safetensor_path)
            lidar_tensors = safetensor_dict['neck_features']

        if "zero_shot" in self.mode:
            instruction = (
                "\nAnalyze the scene and provide your response in two parts. "
                "First, briefly explain your reasoning. Second, output the final decision in a markdown JSON block.\n\n"
                "EXAMPLE FORMAT:\n"
                "Based on the images, I can see X, Y, and Z...\n"
                "```json\n"
                '{"decision": "yield", "hazard_level": "high", "count": 2, '
                '"grounded_objects": [{"type": "pedestrian", "bbox": [10, 20, 30, 40]}]}\n'
                "```\n\n"
                "Now, provide your analysis and JSON for the current images."
            )

            user_query += instruction

        # Format for Qwen
        user_content = [
            {"type": "image", "image": img} for img in images
        ]
        user_content.append({"type": "text", "text": user_query})

        messages = [
            {
                "role": "user",
                "content": user_content
            }
        ]

        # Add response for training
        if self.is_training:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })

        return messages, lidar_tensors, sample['id']