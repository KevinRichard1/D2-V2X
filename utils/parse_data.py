'''Parse annotation json'''
import os
import json
import math
import random
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_DIR = '../data'
OUTPUT_DIR = '../data/processed'
DEBUG = False # Set to True to visualize BEV of first frame in each split

def load_json(file_path):
    '''Load json file'''
    with open(file_path, 'r') as f:
        return json.load(f)

def load_calibrations(calib_dir='../data/calib'):
    '''Load camera dimensions and projection matrices'''
    sensor_map = {
        's110_camera_basler_south1_8mm': 'projection_from_s110_lidar_ouster_south',
        's110_camera_basler_south2_8mm': 'projection_from_s110_lidar_ouster_south',
        's110_camera_basler_north_8mm': 'projection_from_s110_lidar_ouster_north',
        's110_camera_basler_east_8mm': 'projection_from_s110_lidar_ouster_south',
        'vehicle_camera_basler_16mm': 'projection_from_vehicle_lidar_robosense'
    }

    calib_cache = {}
    for cam, preferred_key in sensor_map.items():
        file_path = os.path.join(calib_dir, f"{cam}.json")
        if not os.path.exists(file_path):
            continue

        data = load_json(file_path)

        # Helper to calculate projection matrix if missing
        def get_projection_matrix(intrinsic, R, t):
            K = np.array(intrinsic)
            R = np.array(R)
            t = np.array(t).reshape(3, 1)
            
            extrinsic = np.hstack((R, t))
            
            return K @ extrinsic

        # Calculate projection matrix based on sensor
        p_matrix = data.get(preferred_key)
        if p_matrix is None or (np.all(np.array(p_matrix) == -1) or len(p_matrix) == 0):
            R = data.get('rotation_matrix')
            t = data.get('translation_matrix')
            K = data.get('optimal_intrinsic_camera_matrix')
            
            if R and t and K and not np.all(np.array(R) == -1):
                p_matrix = get_projection_matrix(K, R, t)
            else:
                p_matrix = []
            
        calib_cache[cam] = {
            'width': data.get('image_width', 1920),
            'height': data.get('image_height', 1200),
            'p_matrix': np.array(p_matrix),
            'p_matrix_origin': preferred_key
        }
    return calib_cache

def extract_cuboid(obj_data):
    '''Extract cuboid from object data'''
    val = obj_data['cuboid']['val']

    # Extract center, dimensions, and quaternion
    center = (val[0], val[1], val[2])
    dims = (val[7], val[8], val[9])
    qx, qy, qz, qw = val[3], val[4], val[5], val[6]

    # Calculate yaw
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return {
        'center': center,
        'dims': dims,
        'yaw': yaw
    }

def calculate_occlusion(objects_list):
    '''Calculate if objects are occluded by each other'''
    for obj in objects_list:
        cuboid = extract_cuboid(obj['object_data'])

        x, y, z = cuboid['center']
        l, w, h = cuboid['dims']
        yaw = cuboid['yaw']

        # Calculate distance
        dist = math.sqrt(x**2 + y**2)
        obj['calculated_distance'] = dist

        # Calculate angle of object from sensor
        center_angle = math.degrees(math.atan2(y, x))
        relative_angle = yaw - math.atan2(y, x)

        apparent_width = abs(l * math.sin(relative_angle)) + abs(w * math.cos(relative_angle))
        angular_width = math.degrees(2 * math.atan2(apparent_width, 2 * max(dist, 0.1)))

        obj['angle_min'] = center_angle - (angular_width / 2)
        obj['angle_max'] = center_angle + (angular_width / 2)
        obj['final_occlusion'] = "clear"

    objects_list.sort(key=lambda item: item['calculated_distance'])

    # Check for overlapping shadows
    for i in range(len(objects_list)):
        far_obj = objects_list[i]

        for j in range(i):
            close_obj = objects_list[j]

            if (far_obj['angle_min'] < close_obj['angle_max'] and 
                far_obj['angle_max'] > close_obj['angle_min']):
                
                close_id = close_obj['object_data']['name']
                far_obj['final_occlusion'] = f"occluded_by_{close_id}".lower()
                break
    
    return objects_list

def count_points(pcd_data, cube, DEBUG=False):
    '''Manually count points if not present'''
    try:
        points = pcd_data
        if len(points) == 0: 
            return 0

        center = np.array(cube['center'])
        translated_points = points - center

        # Filter points near the box
        l, w, h = cube['dims']
        max_dist = max(l, w, h) + 1.0
        
        # Calculate distance of translated points from origin
        distances = np.linalg.norm(translated_points, axis=1)
        local_mask = distances < max_dist
        local_points = translated_points[local_mask]
        
        if len(local_points) == 0:
            return 0

        # Rotate points to align with axis
        yaw = cube['yaw']
        rotation_matrix = np.array([
            [np.cos(-yaw), -np.sin(-yaw), 0],
            [np.sin(-yaw),  np.cos(-yaw), 0],
            [0,             0,            1]
        ])
        rotated_points = translated_points @ rotation_matrix.T

        # Filter by dimensions
        l, w, h = cube['dims']
        in_x = np.abs(rotated_points[:, 0]) <= (l/2)
        in_y = np.abs(rotated_points[:, 1]) <= (w/2)
        in_z = np.abs(rotated_points[:, 2]) <= (h/2)

        count = np.sum(in_x & in_y & in_z)

        # Visualize results
        if DEBUG:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            mask = in_x & in_y & in_z

            if(np.any(mask)):
                ax.scatter(xs=rotated_points[mask, 0],
                        ys=rotated_points[mask, 1],
                        zs=rotated_points[mask, 2],
                        s=2, c='blue', label='Inside Box'
                        )
            
            inv_mask = ~mask
            step = max(1, len(rotated_points) // 1000)

            ax.scatter(xs=rotated_points[inv_mask, 0][::step],
                    ys=rotated_points[inv_mask, 1][::step],
                    zs=rotated_points[inv_mask, 2][::step],
                    s=0.5, c='gray', alpha=0.3
            )
            plt.title(f"Point Count: {int(count)}")
            plt.show()

        return int(count)

    except Exception as e:
        print(f"Error counting points for {pcd_data.split('/')[-1]}: {e}")
        return 0
    
def get_3d_corners(cube):
    '''Returns the 8 corners of a 3D cuboid'''
    l, w, h = cube['dims']
    x, y, z = cube['center']
    yaw = cube['yaw']

    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

    # Rotation matrix
    rot_mat = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0,             0,             1]
    ])

    corners_3d = np.dot(rot_mat, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z

    return corners_3d.T

def project_to_2d(corners_3d, cam_data):
    '''Projects 3D corners to 2D image plane and normalizes'''
    p_matrix = cam_data['p_matrix']
    if p_matrix is None or p_matrix.shape != (3, 4):
        return None, False

    corners_hom = np.hstack((corners_3d, np.ones((8, 1))))
    pts_2d_hom = (p_matrix @ corners_hom.T).T
    
    # Filter points behind the camera
    depth = pts_2d_hom[:, 2]
    if np.any(depth <= 0.1): 
        return None, False

    # Normalize by depth
    u = pts_2d_hom[:, 0] / depth
    v = pts_2d_hom[:, 1] / depth

    # Form 2D bounding box
    xmin, xmax = np.min(u), np.max(u)
    ymin, ymax = np.min(v), np.max(v)

    w, h = cam_data['width'], cam_data['height']
    if xmax < 0 or xmin > w or ymax < 0 or ymin > h:
        return None, False
    
    margin = 5
    is_truncated = bool(xmin < margin or xmax > (w - margin) or ymin < margin or ymax > (h - margin))

    # Clamp coordinates
    xmin, xmax = max(0, min(xmin, w)), max(0, min(xmax, w))
    ymin, ymax = max(0, min(ymin, h)), max(0, min(ymax, h))

    norm_box = [
        int((xmin / w) * 1000),
        int((ymin / h) * 1000),
        int((xmax / w) * 1000),
        int((ymax / h) * 1000)
    ]

    if (norm_box[2] - norm_box[0] < 5) or (norm_box[3] - norm_box[1] < 5):
        return None, False
    
    return norm_box, is_truncated

def bin_distance(distance_m):
    '''Convert distance into category'''
    if distance_m < 15.0:
        return "close"
    elif distance_m < 40.0:
        return "mid-range"
    else:
        return "far"

def get_relative_position(x, y, sensor_source):
    '''Maps 3D coordinates to directions'''
    if 'vehicle' in sensor_source:
        if x > 7.0: pos_x = "front"
        elif x < -7.0: pos_x = "rear"
        else: pos_x = "adjacent"
        
        if y > 2.5: pos_y = "left"
        elif y < -2.5: pos_y = "right"
        else: pos_y = "center"
        return f"{pos_x}-{pos_y}"
    else:
        if x > 10.0: pos_x = "East"
        elif x < -10.0: pos_x = "West"
        else: pos_x = "Center-X"
        
        if y > 10.0: pos_y = "North"
        elif y < -10.0: pos_y = "South"
        else: pos_y = "Center-Y"
        
        if pos_x == "Center-X" and pos_y == "Center-Y":
            return "intersection center"
        
        return f"{pos_y} {pos_x}".lower().replace("center-y ", "").replace(" center-x", "").strip()
    
def get_size_category(l, w, h):
    '''Converts bounding box dimensions into natural language sizes'''
    volume = l * w * h
    if volume > 60: return "very large"
    elif volume > 20: return "large"
    elif volume > 8: return "medium"
    elif volume > 2: return "small"
    else: return "very small"

def process_split(split):
    '''Loops through JSON files and processes each in a split'''
    calib_cache = load_calibrations()
    data_path = f"{INPUT_DIR}/{split}/labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
    parsed_frames = []
    corrected_objects = 0

    for file in tqdm(os.listdir(data_path)):
        if file.endswith('.json'):
            data = load_json(os.path.join(data_path, file))

            # Read json info
            frame_id = list(data['openlabel']['frames'].keys())[0]
            content = data['openlabel']['frames'][frame_id]

            # Extract metadata
            timestamp = content['frame_properties']['timestamp']
            
            raw_image_files = content['frame_properties']['image_file_names']
            image_paths = {}
            for img in raw_image_files:
                for cam_name in calib_cache.keys():
                    if cam_name in img:
                        folder_name = cam_name
                        rel_path = f"./data/{split}/images/{folder_name}/{img}"
                        image_paths[cam_name] = rel_path
                        break

            raw_pcd_files = content['frame_properties']['point_cloud_file_names']
            pcd_paths = []
            for pcd in raw_pcd_files:
                parts = pcd.split('_')
                folder_name = "_".join(parts[2:]).replace('.pcd', '')
                rel_path = f"./data/{split}/point_clouds/{folder_name}/{pcd}"
                pcd_paths.append(rel_path)

            registered_pcd = pcd_paths[-1].replace('./', '../')
            if os.path.exists(registered_pcd):
                pcd = o3d.io.read_point_cloud(registered_pcd)
                pcd_data = np.asarray(pcd.points)

            transforms_dict = content.get('frame_properties', {}).get('transforms', {})
            sensor_transform = transforms_dict.get('vehicle_lidar_robosense_to_s110_lidar_ouster_south', {})
            transforms = sensor_transform.get('transform_src_to_dst', {}).get('matrix4x4', [])

            # Calculate occlusions
            objects_list = list(content['objects'].values())
            objects_list = calculate_occlusion(objects_list)

            # Extract object information
            simplified_objects = []
            for obj in objects_list:
                obj_data = obj['object_data']
                cube = extract_cuboid(obj_data)
                dist = obj['calculated_distance']

                # Get sensor source and color
                sensor_source = "unknown"
                color = "unknown"
                for attr in obj_data['cuboid'].get('attributes', {}).get('text', []):
                    if attr['name'] == 'sensor_id':
                        val = attr['val']
                        if val != "":
                            sensor_source = val
                    if attr['name'] == 'body_color':
                        val = attr['val']
                        if val != "":
                            color = val

                # Calculate heading
                yaw_deg = math.degrees(cube['yaw'])
                if 'vehicle' in sensor_source:
                    if -45 < yaw_deg <= 45: heading = "facing forward"
                    elif 45 < yaw_deg <= 135: heading = "facing left"
                    elif -135 < yaw_deg <= -45: heading = "facing right"
                    else: heading = "facing backward"
                else:
                    if -45 < yaw_deg <= 45: heading = "facing east"
                    elif 45 < yaw_deg <= 135: heading = "facing north"
                    elif -135 < yaw_deg <= -45: heading = "facing south"
                    else: heading = "facing west"

                # Calculate density
                num_points = -1
                for attr in obj_data['cuboid'].get('attributes', {}).get('num', []):
                    if attr['name'] == 'num_points':
                        num_points = attr['val']

                        # Count points manually
                        if num_points == -1:
                            num_points = count_points(pcd_data, cube)
                            corrected_objects += 1

                if num_points > 1500: density = "ultra-dense"
                elif num_points > 500: density = "high"
                elif num_points > 100: density = "medium"
                elif num_points > 0:  density = "sparse"    
                elif num_points == 0: density = "trace"
                else: density = "unknown"

                corners_3d = get_3d_corners(cube)
                bboxes_2d = {}
                truncation_info = {}
                best_camera = None
                max_area = 0

                if num_points > 0:
                    for cam_name, cam_data in calib_cache.items():
                        origin = cam_data.get('p_matrix_origin', '')
                        if sensor_source == "unknown" or sensor_source in origin:
                            result = project_to_2d(corners_3d, cam_data)
                            if result is not None and result[0] is not None:
                                proj_result, is_trunc = result
                                bboxes_2d[cam_name] = proj_result
                                truncation_info[cam_name] = is_trunc

                                # Find best view
                                u1, v1, u2, v2 = proj_result
                                area = (u2 - u1) * (v2 - v1)
                                if area > max_area and not is_trunc:
                                    max_area = area
                                    best_camera = cam_name
                if best_camera is None and bboxes_2d:
                    best_camera = max(bboxes_2d, key=lambda k: (bboxes_2d[k][2]-bboxes_2d[k][0]) * (bboxes_2d[k][3]-bboxes_2d[k][1]))

                size_desc = get_size_category(cube['dims'][0], cube['dims'][1], cube['dims'][2])
                position_desc = get_relative_position(cube['center'][0], cube['center'][1], sensor_source)
                is_visible_to_camera = len(bboxes_2d) > 0

                simplified_objects.append({
                    "id": obj_data['name'].lower(),
                    "x": round(cube['center'][0], 2),
                    "y": round(cube['center'][1], 2),
                    "type": obj_data['type'].lower(),
                    "size": size_desc,
                    "distance_category": bin_distance(dist),
                    "distance_m": round(dist, 2),
                    "position": position_desc,
                    "visibility": obj['final_occlusion'],
                    "length_m": round(cube['dims'][0], 2),
                    "width_m": round(cube['dims'][1], 2),
                    "height_m": round(cube['dims'][2], 2),
                    "heading": heading,
                    "detected_by": sensor_source,
                    "density": density,
                    "color": color,
                    "bboxes": bboxes_2d,
                    "is_truncated": truncation_info,
                    "is_visible_in_images": is_visible_to_camera
                })

            # Append clean frame
            parsed_frames.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "images": image_paths,
                "lidar_files": pcd_paths,
                "transforms": transforms,
                "scene_stats": {
                    "total_objects": len(simplified_objects)
                },
                "objects": simplified_objects
            })
    
    return parsed_frames

def save_metrics(data, split):
    '''Save processed list into a JSON file'''
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    save_path = os.path.join(OUTPUT_DIR, f"{split}_metrics.json")

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

    return save_path

def debug_viz_bev(parsed_frames, frame_idx=0):
    '''Debug Visualization of BEV'''
    frame = parsed_frames[frame_idx]
    plt.figure(figsize=(10, 10))
    plt.plot(0, 0, 'rx', markersize=12, label='Sensor Origin')
    
    for obj in frame['objects']:
        x = obj['x']
        y = obj['y']
        
        color_map = {'white': 'gray', 'black': 'black', 'RED': 'red', 'GREEN': 'green', 'unknown': 'blue'}
        obj_color = color_map.get(obj.get('color', 'unknown'), 'blue')
        plt.scatter(x, y, c=obj_color, label=obj['type'])
        
        plt.text(x + 0.5, y + 0.5, f"{obj['type']}\n{obj['distance_m']}m")

    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    print("Processing train files:")
    original_train = process_split('train')

    random.seed(42)
    random.shuffle(original_train)

    # Split train into new train and val
    split_idx = int(len(original_train) * 0.85) # Keep 680 as train, change 120 to val
    new_train = original_train[:split_idx]
    new_val = original_train[split_idx:]

    # Use val files as new test
    print("Processing val files:")
    new_test = process_split('val')

    if DEBUG and len(new_train) > 0 and len(new_val) > 0 and len(new_test) > 0:
        print(f"Visualizing debug BEV for training data...")
        debug_viz_bev(new_train, frame_idx=0)
        print(f"Visualizing debug BEV for validation data...")
        debug_viz_bev(new_val, frame_idx=0)
        print(f"Visualizing debug BEV for testing data...")
        debug_viz_bev(new_test, frame_idx=0)

    save_metrics(new_train, 'train')
    save_metrics(new_val, 'val')
    save_metrics(new_test, 'test')
    print(f"All splits processed and saved to {OUTPUT_DIR}")