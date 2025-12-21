import os
import random
import cv2
import numpy as np
from pathlib import Path
import shutil

def adjust_brightness_contrast(image, brightness_factor=None, contrast_factor=None):
    """
    Adjust brightness and contrast of an image
    brightness_factor: if > 1, increases brightness. if < 1, decreases brightness
    contrast_factor: if > 1, increases contrast. if < 1, decreases contrast
    """
    if brightness_factor is None:
        # Random brightness factor between 0.6 and 1.4
        brightness_factor = random.uniform(0.6, 1.4)
    
    if contrast_factor is None:
        # Random contrast factor between 0.7 and 1.3
        contrast_factor = random.uniform(0.7, 1.3)
    
    # Adjust brightness
    bright_img = cv2.multiply(image, brightness_factor)
    
    # Adjust contrast
    mean = np.mean(bright_img)
    contrast_img = cv2.addWeighted(bright_img, contrast_factor, mean, 0, 0)
    
    # Clip values to valid range [0, 255]
    final_img = np.clip(contrast_img, 0, 255).astype(np.uint8)
    
    return final_img, brightness_factor, contrast_factor

def process_image_and_annotation(image_path, label_path, output_image_path, output_label_path):
    """Process image by changing its exposure/brightness and copy its annotation"""
    # Read image
    img = cv2.imread(str(image_path))
    
    # Adjust brightness and contrast
    adjusted_img, brightness, contrast = adjust_brightness_contrast(img)
    
    # Copy annotation file if it exists (no modification needed for brightness changes)
    if os.path.exists(label_path):
        shutil.copy2(label_path, output_label_path)
    
    # Save adjusted image
    cv2.imwrite(str(output_image_path), adjusted_img)
    
    return brightness, contrast

if __name__ == "__main__":
    pass
