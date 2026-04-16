import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

def create_color_map_by_height(pcd):
    """Colors the point cloud based on the Z-coordinate"""
    points = np.asarray(pcd.points)
    z = points[:, 2]
    
    # Normalize Z values for the colormap
    z_norm = (z - z.min()) / (z.max() - z.min())
    
    # Apply colormap
    cmap = plt.get_cmap("viridis")
    colors = cmap(z_norm)[:, :3]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def capture_high_res_image(geometry, output_path, point_size=2.0):
    """Capture and save image"""
    print(f"Opening visualizer for {output_path}...")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    vis.add_geometry(geometry)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = point_size
    
    vis.run()
    vis.capture_screen_image(output_path)
    vis.destroy_window()
    print(f"Saved: {output_path}\n")

def generate_paper_visuals(pcd_path, output_dir="../data/diagram_assets"):
    """Generates both raw point cloud and voxelized images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    if not pcd.has_points():
        print("Error: Point cloud is empty or invalid.")
        return
        
    # Process and save Raw Point Cloud View
    colored_pcd = create_color_map_by_height(pcd)
    raw_output_path = os.path.join(output_dir, "raw_point_cloud.png")
    capture_high_res_image(colored_pcd, raw_output_path, point_size=3.0)
    
    # Process and save Voxelized View
    voxel_size = 0.5 
    print(f"Generating voxel grid with size {voxel_size}m...")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(colored_pcd, voxel_size=voxel_size)
    
    voxel_output_path = os.path.join(output_dir, "voxel_features.png")
    capture_high_res_image(voxel_grid, voxel_output_path)

if __name__ == "__main__":
    SAMPLE_PCD_PATH = "../data/val/point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/T.pcd" 
    
    if os.path.exists(SAMPLE_PCD_PATH):
        generate_paper_visuals(SAMPLE_PCD_PATH)
    else:
        print(f"File not found: {SAMPLE_PCD_PATH}")