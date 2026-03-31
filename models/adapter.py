import torch
import torch.nn as nn

class LiDARMLP(nn.Module):
    '''MLP to map LiDAR features to Qwen3-VL embedding space'''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Downsampling
        self.pool = nn.AdaptiveAvgPool2d((32, 32))

        # Define MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, lidar_features: torch.Tensor) -> torch.Tensor:
        if lidar_features.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {lidar_features.ndim}D")
        
        # Pool, flatten, and transpose features
        pooled_features = self.pool(lidar_features)
        flattened = pooled_features.flatten(start_dim=2)
        tokens = flattened.transpose(1, 2)

        proj_tensor = self.mlp(tokens)
        return proj_tensor        

def inject_lidar_embeddings(
    model, 
    input_ids: torch.Tensor, 
    lidar_tokens: torch.Tensor, 
    lidar_token_id: int
) -> torch.Tensor:
    '''Converts text placeholder tokens to embeddings'''
    # Get base embeddings
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).clone()

    # Find positions of lidar_token_id
    lidar_mask = (input_ids == lidar_token_id)

    # Ensure number of LiDAR tokens in text matches generated LiDAR feature tokens
    expected_tokens = lidar_mask.sum().item()
    provided_tokens = lidar_tokens.shape[0] * lidar_tokens.shape[1]

    if expected_tokens != provided_tokens:
        raise ValueError(
            f"Token mismatch: Prompt contains {expected_tokens} <lidar> tokens, "
            f"but MLP provided {provided_tokens} LiDAR embeddings."
        )
    
    # Replace base embeddings
    lidar_tokens = lidar_tokens.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    inputs_embeds[lidar_mask] = lidar_tokens.view(-1, lidar_tokens.shape[-1])

    return inputs_embeds