import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from safetensors.numpy import save_file
import feature_extractor as fe

def extract_features(args):
    """Extract features for all samples using TUMTraf JSON structure."""
    print(f"Initializing CenterPoint model from {args.checkpoint}...")
    fe.initialize_model(args)
    print("Model initialized")
    
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON file
    json_path = Path(args.dataset_dir) / args.json_name
    print(f"Loading annotations from: {json_path}")
    
    with open(json_path, 'r') as f:
        samples = json.load(f)
        
    print(f"Found {len(samples)} samples to process.")
    
    results = []
    failed = 0
    
    # Iterate through the TUMTraf samples
    for sample in tqdm(samples, desc=f"Processing {args.mode} samples"):
        sample_id = sample.get('id')
        lidar_rel_path = sample.get('file_metadata', {}).get('lidar_path', '')
        clean_rel_path = lidar_rel_path.lstrip('./')
        lidar_path = Path(args.data_root) / clean_rel_path
        output_file = output_dir / f"{sample_id}{args.suffix}.safetensors"
        
        if output_file.exists():
            continue

        if not lidar_path.exists():
            print(f"File not found: {lidar_path}")
            failed += 1
            continue
            
        try:
            # Load point cloud and extract features
            points = fe.load_point_cloud(str(lidar_path))

            # Filter NaNs
            initial_count = len(points)
            points = points[~np.isnan(points).any(axis=1)]
            final_count = len(points)

            if initial_count != final_count:
                print(f"Cleaned {initial_count - final_count} NaN points from sample {sample_id}")

            if len(points) == 0:
                print(f"Skipping {sample_id}: No valid points after cleaning.")
                continue

            # Extract features from cleaned points
            features = fe.run_model(points, fp16=args.fp16)
            
            # Convert tensors to numpy arrays
            features_np = {}
            for key, value in features.items():
                if hasattr(value, 'cpu'):
                    features_np[key] = value.cpu().numpy()
                else:
                    features_np[key] = value
            
            metadata = {
                'sample_id': str(sample_id),
                'lidar_path': str(lidar_path),
                'num_points': str(points.shape[0])
            }
            
            # Save tensors
            save_file(features_np, output_file, metadata=metadata)
            
            results.append({
                'sample_id': sample_id,
                'num_points': points.shape[0]
            })
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            failed += 1
            continue
            
    # Save summary
    summary = {
        'total_processed': len(results),
        'failed_samples': failed,
        'output_dir': str(output_dir)
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Extracted features for {len(results)} samples")
    print(f"Failed: {failed} samples")
    print(f"Saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features for TUMTraf using CenterPoint")
    
    # Dataset args
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--json_name', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default="./data/datasets", 
                        help="Directory containing the JSON files")
    parser.add_argument('--data_root', type=str, default="./", 
                        help="Base directory that the JSON paths are relative to")
    parser.add_argument('--output_dir', type=str, default='./data/tumtraf_features')
    parser.add_argument('--suffix', type=str, default="", help="Suffix for output filename (e.g., _ego)")
    
    # Model args
    parser.add_argument('--config', type=str, default='CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py')
    parser.add_argument('--checkpoint', type=str, default='CenterPoint/epoch_20.pth')
    parser.add_argument('--fp16', action='store_true', help="Use half precision")
    
    args = parser.parse_args()
    extract_features(args)