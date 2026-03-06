'''Parse annotation json'''
import os
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt

SPLITS = ['train', 'val', 'test']
INPUT_DIR = '../data'
OUTPUT_DIR = '../data/processed'
DEBUG = False # Set to True to visualize BEV of first frame in each split

def load_json(file_path):
    '''Load json file'''
    with open(file_path, 'r') as f:
        return json.load(f)

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
        obj['final_occlusion'] = "CLEAR"

    objects_list.sort(key=lambda item: item['calculated_distance'])

    # Check for overlapping shadows
    for i in range(len(objects_list)):
        far_obj = objects_list[i]

        for j in range(i):
            close_obj = objects_list[j]

            if (far_obj['angle_min'] < close_obj['angle_max'] and 
                far_obj['angle_max'] > close_obj['angle_min']):
                
                close_type = close_obj['object_data']['type']
                far_obj['final_occlusion'] = f"OCCLUDED_BY_{close_type}"
                break
    
    return objects_list

def bin_distance(distance_m):
    '''Convert distance into category'''
    if distance_m < 15.0:
        return "close"
    elif distance_m < 40.0:
        return "mid-range"
    else:
        return "far"

def get_relative_position(x, y):
    '''Maps 3D coordinates to directions'''
    if x > 7.0:
        pos_x = "front"
    elif x < -7.0:
        pos_x = "rear"
    else:
        pos_x = "adjacent"
    
    if y > 2.5:
        pos_y = "left"
    elif y < -2.5:
        pos_y = "right"
    else:
        pos_y = "center"

    return f"{pos_x}-{pos_y}"

def process_split(split):
    '''Loops through JSON files and processes each in a split'''
    data_path = os.path.join(INPUT_DIR, split, 'labels_point_clouds', 's110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')

    parsed_frames = []

    for file in os.listdir(data_path):
        if file.endswith('.json'):
            data = load_json(os.path.join(data_path, file))

            # Read json info
            frame_id = list(data['openlabel']['frames'].keys())[0]
            content = data['openlabel']['frames'][frame_id]

            # Extract metadata
            timestamp = content['frame_properties']['timestamp']
            
            raw_image_files = content['frame_properties']['image_file_names']
            image_paths = []
            for img in raw_image_files:
                parts = img.split('_')
                folder_name = "_".join(parts[2:]).replace('.jpg', '')
                rel_path = f"./data/{split}/images/{folder_name}/{img}"
                image_paths.append(rel_path)

            raw_pcd_files = content['frame_properties']['point_cloud_file_names']
            pcd_paths = []
            for pcd in raw_pcd_files:
                parts = pcd.split('_')
                folder_name = "_".join(parts[2:]).replace('.pcd', '')
                rel_path = f"./data/{split}/point_clouds/{folder_name}/{pcd}"
                pcd_paths.append(rel_path)

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
                        sensor_source = attr['val']
                    if attr['name'] == 'body_color':
                        color = attr['val']
                        if color == "":
                            color = "unknown"

                # Calculate heading
                yaw_deg = math.degrees(cube['yaw'])
                if -45 < yaw_deg <= 45: heading = "facing forward"
                elif 45 < yaw_deg <= 135: heading = "facing left"
                elif -135 < yaw_deg <= -45: heading = "facing right"
                else: heading = "facing backward"

                # Calculate density
                num_points = 0
                for attr in obj_data['cuboid'].get('attributes', {}).get('num', []):
                    if attr['name'] == 'num_points':
                        num_points = attr['val']

                if num_points > 1500:
                    density = "ultra-dense"
                elif num_points > 500:
                    density = "high"
                elif num_points > 100:
                    density = "medium"
                elif num_points > 0: 
                    density = "sparse"    
                else:
                    density = "trace"

                simplified_objects.append({
                    "id": obj_data['name'],
                    "x": cube['center'][0],
                    "y": cube['center'][1],
                    "type": obj_data['type'],
                    "distance_category": bin_distance(dist),
                    "distance_m": round(dist, 2),
                    "position": get_relative_position(cube['center'][0], cube['center'][1]),
                    "visibility": obj['final_occlusion'],
                    "length_m": round(cube['dims'][0], 2),
                    "width_m": round(cube['dims'][1], 2),
                    "height_m": round(cube['dims'][2], 2),
                    "heading": heading,
                    "detected_by": sensor_source,
                    "density": density,
                    "color": color
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
    for split in SPLITS:
        print(f"Processing {split} split...")
        processed_data = process_split(split)

        if DEBUG and len(processed_data) > 0:
            print(f"Visualizing debug BEV for {split}...")
            debug_viz_bev(processed_data, frame_idx=0)

        save_path = save_metrics(processed_data, split)
        print(f"Saved to {save_path}")