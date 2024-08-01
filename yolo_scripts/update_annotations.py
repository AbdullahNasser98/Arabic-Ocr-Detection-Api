import os
import json
import shutil
from pathlib import Path
import cv2
import random
from PIL import Image, ExifTags

# Define paths
source_dir = '/mnt/d/valify/data'
target_dir = '/mnt/d/valify/yolo_data'
os.makedirs(target_dir, exist_ok=True)
os.makedirs(f"{target_dir}/images/train", exist_ok=True)
os.makedirs(f"{target_dir}/images/val", exist_ok=True)
os.makedirs(f"{target_dir}/labels/train", exist_ok=True)
os.makedirs(f"{target_dir}/labels/val", exist_ok=True)

# Define class names
class_names = ['Title', 'Body text', 'Page']

# Function to convert polygon points to bounding box
def get_bounding_box(points):
    x_coordinates = [point[0] for point in points]
    y_coordinates = [point[1] for point in points]
    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)
    return x_min, y_min, x_max, y_max

# Function to normalize bounding box coordinates
def normalize_bbox(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Function to convert image to .jpg and handle orientation
def convert_to_jpg(img_path, save_path):
    try:
        with Image.open(img_path) as img:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None:
                orientation = exif.get(orientation)
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
            rgb_img = img.convert('RGB')
            rgb_img.save(save_path, format='JPEG')
    except Exception as e:
        print(f"Error converting image {img_path}: {e}")

# Iterate over each sample directory
for sample_dir in os.listdir(source_dir):
    ann_dir = os.path.join(source_dir, sample_dir, 'ann')
    img_dir = os.path.join(source_dir, sample_dir, 'img')
    print(sample_dir)
    
    for ann_file in os.listdir(ann_dir):
        with open(os.path.join(ann_dir, ann_file)) as f:
            data = json.load(f)
        
        image_file = ann_file.replace('.json', '')
        img_path = None
        
        for ext in ['.jpg', '.jpeg', '.png']:
            if os.path.exists(os.path.join(img_dir, image_file + ext)):
                img_path = os.path.join(img_dir, image_file + ext)
                break
        
        if img_path is None:
            continue  # Skip if image is not found
        
        new_image_file = image_file + f'_{sample_dir}.jpg'
        
        # Split data into train and validation
        if random.random() < 0.8:  # 80% for training
            target_img_dir = f"{target_dir}/images/train"
            target_label_dir = f"{target_dir}/labels/train"
        else:  # 20% for validation
            target_img_dir = f"{target_dir}/images/val"
            target_label_dir = f"{target_dir}/labels/val"
        
        # Convert and copy image to target directory
        new_img_path = os.path.join(target_img_dir, new_image_file)
        convert_to_jpg(img_path, new_img_path)
        
        img = cv2.imread(new_img_path)
        if img is None:
            print(f"Error loading image: {new_img_path}")
            continue
        img_height, img_width = img.shape[:2]
        
        # Prepare annotation file
        label_file = ann_file.replace('.json', f'_{sample_dir}.txt')
        with open(os.path.join(target_label_dir, label_file), 'w') as lf:
            for obj in data['objects']:
                class_id = class_names.index(obj['classTitle'])
                points = obj['points']['exterior']
                x_min, y_min, x_max, y_max = get_bounding_box(points)
                x_center, y_center, width, height = normalize_bbox(x_min, y_min, x_max, y_max, img_width, img_height)
                lf.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Create the YAML configuration file
yaml_content = f"""
train: {target_dir}/images/train
val: {target_dir}/images/val

nc: {len(class_names)}
names: {class_names}
"""

with open(f"{target_dir}/dataset.yaml", 'w') as yaml_file:
    yaml_file.write(yaml_content)

print("Dataset is prepared for YOLO training.")
