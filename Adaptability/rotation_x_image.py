import os
import random
import cv2
import numpy as np
from pathlib import Path
import shutil

"""
Tilt images around the X axis (vertical perspective tilt).
This script performs a perspective transform that simulates rotating the image
around the X axis (tilting forward/back). It also updates YOLO-format annotations
by transforming the four corner points of each bounding box and recomputing the
axis-aligned bounding box in normalized coordinates.
"""


def build_tilt_transform_matrix(w, h, tilt_deg):
    """Return a 3x3 perspective transform matrix that simulates an X-axis tilt.
    tilt_deg: positive tilts the top edge away (makes top narrower), negative brings it closer.
    """
    # Limit tilt to avoid degenerate transforms
    max_tilt = 15  # reduced from 60 to 15 degrees
    tilt_deg = np.clip(tilt_deg, -max_tilt, max_tilt)

    # Strength parameter derived from angle; map degrees to a vertical shift factor
    t = np.tan(np.radians(tilt_deg)) * 0.5  # scale down to reasonable perspective

    # Define source points (corners of the image)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    # Calculate how much to move top edge inward (toward center) based on t
    dx = t * w
    # Move top-left and top-right towards center by dx
    dst_top_left = np.array([dx, 0])
    dst_top_right = np.array([w - 1 - dx, 0])

    # Optionally scale vertical positions to simulate foreshortening
    # y_scale reduces the height perceived for the top when tilting away
    y_scale = 1.0 - abs(t) * 0.5
    dst_bottom_right = np.array([w - 1, (h - 1) * y_scale + (1 - y_scale) * h * 0.5])
    dst_bottom_left = np.array([0, (h - 1) * y_scale + (1 - y_scale) * h * 0.5])

    dst = np.vstack([dst_top_left, dst_top_right, dst_bottom_right, dst_bottom_left]).astype(np.float32)

    # Compute perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    return M


def transform_point(pt, M):
    """Apply homography M to a single point (x, y)."""
    x, y = pt
    vec = np.array([x, y, 1.0], dtype=np.float32)
    tx, ty, tz = M.dot(vec)
    if tz == 0:
        return (0, 0)
    return (tx / tz, ty / tz)


def yolo_to_corners(xc, yc, w, h, img_w, img_h):
    """Convert YOLO normalized bbox to pixel corner coordinates (4 corners)."""
    x_center = xc * img_w
    y_center = yc * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = x_center - bw / 2.0
    y1 = y_center - bh / 2.0
    x2 = x_center + bw / 2.0
    y2 = y_center + bh / 2.0
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def corners_to_yolo(corners, img_w, img_h, class_id=0):
    """Convert transformed corners (pixel coords) to YOLO normalized bbox (class, xc, yc, w, h)."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    # Clamp to image bounds
    x_min = max(0, min(x_min, img_w - 1))
    x_max = max(0, min(x_max, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    y_max = max(0, min(y_max, img_h - 1))

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0:
        return None  # invalid box after transform

    xc = (x_min + x_max) / 2.0 / img_w
    yc = (y_min + y_max) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    return [class_id, xc, yc, nw, nh]


def process_image_and_annotation(image_path, label_path, output_image_path, output_label_path, tilt_deg):
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return False
    h, w = img.shape[:2]

    # Build transform matrix and apply
    M = build_tilt_transform_matrix(w, h, tilt_deg)
    tilted = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    # Process annotation file if it exists
    transformed_annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            annotations = [line.strip() for line in f if line.strip()]
        for ann in annotations:
            parts = ann.split()
            class_id = int(parts[0])
            vals = list(map(float, parts[1:5]))
            corners = yolo_to_corners(vals[0], vals[1], vals[2], vals[3], w, h)
            transformed_corners = [transform_point(c, M) for c in corners]
            new_box = corners_to_yolo(transformed_corners, w, h, class_id)
            if new_box is not None:
                transformed_annotations.append(new_box)
    else:
        # No label file; that's okay
        pass

    # Write output image
    cv2.imwrite(str(output_image_path), tilted)

    # Write labels
    if transformed_annotations:
        with open(output_label_path, 'w') as f:
            for ann in transformed_annotations:
                f.write(' '.join(map(str, ann)) + '\n')

    return True


def main():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    dataset_root = project_root / 'dataset'
    images_dir = dataset_root / 'images' / 'train'
    labels_dir = dataset_root / 'labels' / 'train'

    print(f"Looking for images in: {images_dir.absolute()}")
    if not images_dir.exists():
        print(f"Error: Images directory does not exist: {images_dir.absolute()}")
        return

    output_base = dataset_root / 'images' / 'train_tilted_x'
    output_labels_base = dataset_root / 'labels' / 'train_tilted_x'

    # Remove old directories if they exist
    if output_base.exists():
        print(f"Removing existing output directory: {output_base}")
        shutil.rmtree(output_base)
    if output_labels_base.exists():
        print(f"Removing existing output directory: {output_labels_base}")
        shutil.rmtree(output_labels_base)

    output_base.mkdir(parents=True, exist_ok=True)
    output_labels_base.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob('*.jpg'))
    print(f"Found {len(image_files)} .jpg files")
    selected_images = random.sample(image_files, min(200, len(image_files)))

    for img_path in selected_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        tilt_deg = random.uniform(-30, 30)  # tilt between -30 and 30 degrees
        output_image_path = output_base / f"{img_path.stem}_tiltx{int(tilt_deg)}.jpg"
        output_label_path = output_labels_base / f"{img_path.stem}_tiltx{int(tilt_deg)}.txt"
        ok = process_image_and_annotation(img_path, label_path, output_image_path, output_label_path, tilt_deg)
        if ok:
            print(f"Processed {img_path.name} tilt={tilt_deg:.2f}")

if __name__ == '__main__':
    main()
