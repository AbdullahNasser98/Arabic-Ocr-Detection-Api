import cv2
import json
import numpy as np



def get_bounding_box(points):
    x_coordinates = [point[0] for point in points]
    y_coordinates = [point[1] for point in points]
    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)
    return x_min, y_min, x_max, y_max

def plot_original_annotations():
    # Load the JSON data
    with open(r'/mnt/d/valify/data/Comics/ann/001.json') as f:
        data = json.load(f)
    # Load the image
    image = cv2.imread(r'/mnt/d/valify/data/Comics/img/001.jpeg')

    for obj in data['objects']:
        class_name = obj['classTitle']
        points = obj['points']['exterior']
        x_min, y_min, x_max, y_max = get_bounding_box(points)
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Put the class name near the bounding box
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    cv2.imwrite('output_image.jpg', image)


def plot_yolo_annotations(image_path, annotations_path, class_names):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Read the YOLO annotations
    with open(annotations_path, 'r') as file:
        annotations = file.readlines()
    
    img_height, img_width, _ = image.shape

    for annotation in annotations:
        # Split the annotation line into components
        components = annotation.strip().split()
        class_id = int(components[0])
        x_center, y_center, width, height = map(float, components[1:])
        
        # Convert normalized coordinates to pixel values
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Calculate the top-left and bottom-right corners of the bounding box
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        
        # Draw the bounding box and label on the image
        label = class_names[class_id]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # # Display the image with annotations
    # cv2.imshow('Image with Annotations', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # Optionally, save the image with bounding boxes
    cv2.imwrite('output_image2.jpg', image)

# Example usage
image_path = '/mnt/d/valify/yolo_data/images/train/001_Comics.jpg'
annotations_path = '/mnt/d/valify/yolo_data/labels/train/001_Comics.txt'
class_names = ['Title', 'Body text', 'Page']  # Adjust based on your classes

plot_yolo_annotations(image_path, annotations_path, class_names)
plot_original_annotations()