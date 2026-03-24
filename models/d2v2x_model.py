import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration
from .adapter import LiDARMLP, inject_lidar_embeddings

class D2V2XModel(nn.Module):
    '''Wrapper class for the D2-V2X architecture'''

    def __init__(self, model_path: str, mode):
        '''Initializes the base VLM and the custom LiDAR adapter'''
        super().__init__()
        
        self.mode = mode
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True
        )

        # Initialize adapter if needed
        if self.mode in ['ego', 'v2x']:
            text_out = self.model.config.text_config.hidden_size

            self.lidar_mlp = LiDARMLP(
                input_dim=512, # From CenterPoint features
                hidden_dim=2048,
                output_dim=text_out
            )

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
        images: list = None, 
        lidar_features: torch.Tensor = None, 
        **kwargs
    ):
        '''Inference pass'''
        if lidar_features is None:
            # Image only text generation
            return self.model.generate(
                input_ids=input_ids,
                images=images,
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

            # Text generation with LiDAR embeddings
            return self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=kwargs.get('attention_mask', None),
                images=images,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")