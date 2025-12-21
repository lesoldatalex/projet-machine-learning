# Ce fichier a été archivé et déplacé dans `archive/Adaptability/change_exposition.py`
# Utiliser `from src.augment.change_exposition import process_image_and_annotation` à la place

raise RuntimeError('Fichier archivé; voir archive/Adaptability/change_exposition.py')

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
    output_base = dataset_root / 'images' / 'train_luminosity'
    output_labels_base = dataset_root / 'labels' / 'train_luminosity'
    
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
    selected_images = random.sample(image_files, min(200, len(image_files)))
    
    # Process each selected image
    for img_path in selected_images:
        # Get corresponding label path
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Create output paths with brightness indicator
        output_image_path = output_base / f"{img_path.stem}_adjusted.jpg"
        output_label_path = output_labels_base / f"{img_path.stem}_adjusted.txt"
        
        # Process the image and its annotation
        brightness, contrast = process_image_and_annotation(
            img_path,
            label_path,
            output_image_path,
            output_label_path
        )
        print(f"Processed {img_path.name} (brightness: {brightness:.2f}, contrast: {contrast:.2f})")

if __name__ == "__main__":
    main()
    print("FINISHED")
