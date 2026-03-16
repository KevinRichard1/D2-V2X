'''Feature extraction from LAS/point cloud files using CenterPoint VoxelNet
   Adapted from CenterPoint/tools/simple_inference_waymo.py'''
import sys
import os
from pathlib import Path
centerpoint_path = Path(__file__).parent / "CenterPoint"
if str(centerpoint_path) not in sys.path:
    sys.path.insert(0, str(centerpoint_path))

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.torchie.trainer import load_checkpoint
from det3d.models import build_detector
from det3d.torchie import Config
import numpy as np
import torch
import open3d as o3d

# Global variables for model and voxel generator
voxel_generator = None 
model = None 
device = None 

def initialize_model(args):
    """Initialize the CenterPoint VoxelNet model with pretrained weights."""
    global model, voxel_generator, device

    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    if args.fp16:
        print("cast model to fp16")
        model = model.half()

    model = model.cuda()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize voxel generator from config
    range = cfg.voxel_generator.range
    voxel_size = cfg.voxel_generator.voxel_size
    max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
    max_voxel_num = cfg.voxel_generator.max_voxel_num[1]
    
    voxel_generator = VoxelGenerator(
        voxel_size=voxel_size,
        point_cloud_range=range,
        max_num_points=max_points_in_voxel,
        max_voxels=max_voxel_num
    )
    
    return model 

def load_point_cloud(file_path):
    """Load point cloud from .las or .pcd.bin file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pcd':            
            try:
                pcd = o3d.io.read_point_cloud(file_path)
            except Exception as e:
                raise ValueError(f"Failed to read PCD file {file_path}: {str(e)}")
            
            # Extract x, y, z, intensity
            points = np.asarray(pcd.points)
            
            # Check if intensity field exists
            if pcd.has_colors():
                intensity = np.asarray(pcd.colors, dtype=np.float32)[:, 0:1] 
            elif hasattr(pcd, 'point') and 'intensity' in pcd.point:
                intensity = pcd.point['intensity'].numpy().astype(np.float32)
            else:
                intensity = np.full((points.shape[0], 1), 0.5, dtype=np.float32)

                if intensity.ndim == 1:
                    intensity = intensity.reshape(-1, 1)
            
            # Stack into (N, 4) array
            pcd_combined = np.hstack((points, intensity))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .las, .bin, .pcd.bin")
        
        # Validate that we have data
        if pcd_combined.shape[0] == 0:
            raise ValueError(f"Point cloud file {file_path} is empty (0 points)")
        
        # Pad with zeros
        if pcd_combined.shape[1] == 4:
            padding = np.zeros((pcd_combined.shape[0], 1), dtype=pcd_combined.dtype)
            pcd_combined = np.hstack([pcd_combined, padding])
        
        return pcd_combined
        
    except Exception as e:
        raise ValueError(f"Unexpected error loading point cloud from {file_path}: {str(e)}")

def _process_inputs(points, fp16):
    """Convert point cloud to voxelized tensors for model input."""
    voxels, coords, num_points = voxel_generator.generate(points)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int32)
    grid_size = voxel_generator.grid_size
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
    num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=device)

    if fp16:
        voxels = voxels.half()

    # Convert points to tensor for the model
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    # CenterPoint expects this specific format
    inputs = dict(
        voxels=voxels,
        num_points=num_points,
        num_voxels=num_voxels,
        coordinates=coords,
        shape=[grid_size],
        points=[points_tensor]
    )

    return inputs

def run_model(points, fp16=False):
    """Extract features from point cloud using the CenterPoint VoxelNet model."""
    # Dictionary to store captured features
    features = {}
    
    # Define hook functions to capture outputs
    def get_reader_hook(name):
        def hook(module, input, output):
            # Handle both single tensors and tuples
            if isinstance(output, tuple):
                features[name] = output[0].detach() if hasattr(output[0], 'detach') else output[0]
            else:
                features[name] = output.detach() if hasattr(output, 'detach') else output
        return hook
    
    def get_backbone_hook(name):
        def hook(module, input, output):
            # Handle both single tensors and tuples
            if isinstance(output, tuple):
                features[name] = output[0].detach() if hasattr(output[0], 'detach') else output[0]
            else:
                features[name] = output.detach() if hasattr(output, 'detach') else output
        return hook
    
    def get_neck_hook(name):
        def hook(module, input, output):
            # Handle both single tensors and tuples
            if isinstance(output, tuple):
                features[name] = output[0].detach() if hasattr(output[0], 'detach') else output[0]
            else:
                features[name] = output.detach() if hasattr(output, 'detach') else output
        return hook
    
    # Register forward hooks on model components
    hook_handles = []
    
    # Hook on reader (voxel feature extractor)
    if hasattr(model, 'reader'):
        handle = model.reader.register_forward_hook(get_reader_hook('reader_features'))
        hook_handles.append(handle)
    
    # Hook on backbone (3D sparse convolution)
    if hasattr(model, 'backbone'):
        handle = model.backbone.register_forward_hook(get_backbone_hook('backbone_features'))
        hook_handles.append(handle)
    
    # Hook on neck (RPN)
    if hasattr(model, 'neck'):
        handle = model.neck.register_forward_hook(get_neck_hook('neck_features'))
        hook_handles.append(handle)
    
    try:
        # Run inference
        with torch.no_grad():
            data_dict = _process_inputs(points, fp16)

            # Reader
            if hasattr(model, 'reader'):
                voxel_features = model.reader(
                    data_dict['voxels'],
                    data_dict['num_points']
                )
                features['reader_features'] = voxel_features
            
            # Backbone
            if hasattr(model, 'backbone'):
                # Prepare input for backbone
                batch_size = 1
                input_shape = data_dict['shape'][0]
                backbone_output = model.backbone(
                    voxel_features,
                    data_dict['coordinates'],
                    batch_size,
                    input_shape
                )
                
                # Handle tuple output from backbone
                if isinstance(backbone_output, tuple):
                    spatial_features = backbone_output[0]
                else:
                    spatial_features = backbone_output
                
                features['backbone_features'] = spatial_features
            
            # Neck
            if hasattr(model, 'neck'):
                if isinstance(spatial_features, dict):
                    # Handle dict output from backbone
                    neck_features = model.neck(spatial_features)
                else:
                    # Handle tensor output
                    neck_features = model.neck(spatial_features)
                
                # Handle tuple output from neck
                if isinstance(neck_features, tuple):
                    neck_features = neck_features[0]
                
                features['neck_features'] = neck_features
        
        # Remove hooks after inference
        for handle in hook_handles:
            handle.remove()
        
        return features
    
    except Exception as e:
        # Clean up hooks in case of error
        for handle in hook_handles:
            handle.remove()
        raise e