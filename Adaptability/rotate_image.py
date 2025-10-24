import os
import random
import cv2
import numpy as np
from pathlib import Path
import shutil

def rotate_point(x, y, angle, cx=0.5, cy=0.5):
    """Rotate a point around a center point
    x, y: normalized coordinates (0-1)
    angle: angle in degrees
    cx, cy: center point (default is image center 0.5, 0.5)
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Translate point to origin
    x_shifted = x - cx
    y_shifted = y - cy
    
    # Rotate point
    x_rotated = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
    y_rotated = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
    
    # Translate back
    x_final = x_rotated + cx
    y_final = y_rotated + cy
    
    return x_final, y_final

def rotate_yolo_annotation(annotation, angle):
    """Rotate YOLO format annotation
    annotation: list containing [class_id, x_center, y_center, width, height]
    angle: angle in degrees
    """
    class_id = annotation[0]
    x_center, y_center = annotation[1], annotation[2]
    width, height = annotation[3], annotation[4]
    
    # Convert to corners (normalized coordinates)
    # Calculate the four corners of the bounding box
    x1 = x_center - width/2
    y1 = y_center - height/2
    x2 = x_center + width/2
    y2 = y_center - height/2
    x3 = x_center + width/2
    y3 = y_center + height/2
    x4 = x_center - width/2
    y4 = y_center + height/2
    
    # Rotate each corner
    corners = [
        rotate_point(x1, y1, angle),
        rotate_point(x2, y2, angle),
        rotate_point(x3, y3, angle),
        rotate_point(x4, y4, angle)
    ]
    
    # Find the new bounding box that encompasses all rotated corners
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    
    # New center point
    new_x_center = (min(xs) + max(xs)) / 2
    new_y_center = (min(ys) + max(ys)) / 2
    
    # New width and height
    new_width = max(xs) - min(xs)
    new_height = max(ys) - min(ys)
    
    # Ensure coordinates and dimensions stay within bounds [0, 1]
    new_x_center = np.clip(new_x_center, 0, 1)
    new_y_center = np.clip(new_y_center, 0, 1)
    new_width = min(new_width, 1.0)
    new_height = min(new_height, 1.0)
    
    return [class_id, new_x_center, new_y_center, new_width, new_height]

def process_image_and_annotation(image_path, label_path, output_image_path, output_label_path, angle):
    """Process both image and its annotation file"""
    # Read and rotate image
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    # Process annotation file if it exists
    if os.path.exists(label_path):
        print("path exist");
        with open(label_path, 'r') as f:
            annotations = f.readlines()
        
        rotated_annotations = []
        for ann in annotations:
            # Convert string annotation to list of floats
            ann_parts = list(map(float, ann.strip().split()))
            # Rotate annotation
            rotated_ann = rotate_yolo_annotation(ann_parts, angle)
            # Convert back to string format
            rotated_annotations.append(' '.join(map(str, rotated_ann)))
        
        # Write rotated annotations
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(rotated_annotations))
    
    # Save rotated image
    cv2.imwrite(str(output_image_path), rotated_img)

def main():
    # Set paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    dataset_root = project_root / 'dataset'
    images_dir = dataset_root / 'images' / 'train'
    labels_dir = dataset_root / 'labels' / 'train'
    
    print(f"Looking for images in: {images_dir.absolute()}")
    
    # Check if directory exists
    if not images_dir.exists():
        print(f"Error: Images directory does not exist: {images_dir.absolute()}")
        return
        
    # Create output directories
    output_base = dataset_root / 'images' / 'train_rotated'
    output_labels_base = dataset_root / 'labels' / 'train_rotated'
    
    # Remove old directories if they exist
    if output_base.exists():
        print(f"Removing existing output directory: {output_base}")
        shutil.rmtree(output_base)
    if output_labels_base.exists():
        print(f"Removing existing output directory: {output_labels_base}")
        shutil.rmtree(output_labels_base)
    
    # Create fresh directories
    print("Creating new output directories...")
    output_base.mkdir(parents=True, exist_ok=True)
    output_labels_base.mkdir(parents=True, exist_ok=True)
    
    # Get list of all images
    image_files = list(images_dir.glob('*.jpg'))
    print(f"Found {len(image_files)} .jpg files")

    # Randomly select 200 images
    selected_images = random.sample(image_files, min(350, len(image_files)))
    print(selected_images);
    
    # Process each selected image
    for img_path in selected_images:
        # Get corresponding label path
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Generate random angle between -15 and +15 degrees
        angle = random.uniform(-15, 15)
        
        # Create output paths
        output_image_path = output_base / f"{img_path.stem}_rot{angle}.jpg"
        output_label_path = output_labels_base / f"{img_path.stem}_rot{angle}.txt"
        
        # Process the image and its annotation
        process_image_and_annotation(
            img_path,
            label_path,
            output_image_path,
            output_label_path,
            angle
        )
        print(f"Processed {img_path.name} with {angle}° rotation")

if __name__ == "__main__":
    main()
    print("FINI")
