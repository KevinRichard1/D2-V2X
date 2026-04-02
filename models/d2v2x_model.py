import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import Qwen3VLForConditionalGeneration
from .adapter import LiDARMLP, inject_lidar_embeddings

class D2V2XModel(nn.Module):
    '''Wrapper class for the D2-V2X architecture'''

    def __init__(self, base_model_path: str, adapter_path:str, mode, quantization_config=None):
        '''Initializes the base VLM and the custom LiDAR adapter'''
        super().__init__()
        
        self.mode = mode
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            attn_implementation="sdpa",
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16
        )

        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading LoRA adapters from {adapter_path}...")
            self.model.load_adapter(adapter_path)

        # Initialize MLP if needed
        if self.mode in ['ego', 'v2x']:
            text_out = self.model.config.text_config.hidden_size

            self.lidar_mlp = LiDARMLP(
                input_dim=512, # From CenterPoint features
                hidden_dim=2048,
                output_dim=text_out
            )

            # Load pretrained weights
            mlp_path = os.path.join(os.path.dirname(adapter_path), "lidar_mlp.safetensors")
            
            if os.path.exists(mlp_path):
                print(f"Loading LiDAR MLP weights from {mlp_path}...")
                mlp_state_dict = load_file(mlp_path)
                self.lidar_mlp.load_state_dict(mlp_state_dict)
            else:
                print(f"WARNING: No LiDAR MLP weights found at {mlp_path}")

            self.lidar_mlp.to(
                device=self.model.device, 
                dtype=self.model.dtype
            )

    def get_input_embeddings(self):
        '''Exposes the base model's embedding layer.'''
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        '''Allows libraries to update the embedding layer.'''
        self.model.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, **kwargs):
        '''Passes gradient checkpointing flag to base model'''
        self.model.gradient_checkpointing_enable(**kwargs)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None, 
        images: list = None, 
        lidar_features: torch.Tensor = None, 
        **kwargs
    ):
        '''Main training pass'''
        if lidar_features is None:
            # Call base model directly
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                labels=labels,
                **kwargs
            )
        elif self.mode in ['ego', 'v2x']:
            # Get LiDAR tokens
            lidar_features = lidar_features.to(
                device=self.model.device, 
                dtype=self.model.dtype
            )
            lidar_tokens = self.lidar_mlp(lidar_features)

            # Add LiDAR embeddings
            inputs_embeds = inject_lidar_embeddings(
                model=self.model,
                input_ids=input_ids,
                lidar_tokens=lidar_tokens,
                lidar_token_id=151669
            )

            # Call base model
            return self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                images=images,
                labels=labels,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        lidar_features: torch.Tensor = None, 
        **kwargs
    ):
        '''Inference pass'''
        if lidar_features is None:
            # Image only text generation
            return self.model.generate(
                input_ids=input_ids,
                **kwargs
            )
        elif self.mode in ['ego', 'v2x']:
            # Get LiDAR tokens
            lidar_features = lidar_features.to(
                device=self.model.device, 
                dtype=self.model.dtype
            )
            lidar_tokens = self.lidar_mlp(lidar_features)

            # Add LiDAR embeddings
            inputs_embeds = inject_lidar_embeddings(
                model=self.model,
                input_ids=input_ids,
                lidar_tokens=lidar_tokens,
                lidar_token_id=151669
            )

            kwargs.pop('input_ids', None)

            # Text generation with LiDAR embeddings
            return self.model.generate(
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")