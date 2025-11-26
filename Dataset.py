
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

# -----------------------------
# 1. Utility functions
# -----------------------------
def yolo_to_boxes(yolo_file, img_width, img_height):
    boxes = []
    class_ids = []
    if not os.path.exists(yolo_file):
        return boxes, class_ids
    with open(yolo_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id, x_center, y_center, w, h = map(float, parts)
            except ValueError:
                print(f"Skipping malformed line in {yolo_file}: '{line.strip()}'")
                continue

            x_center *= img_width
            y_center *= img_height
            w *= img_width
            h *= img_height
            xmin = max(x_center - w/2, 0)
            ymin = max(y_center - h/2, 0)
            xmax = min(x_center + w/2, img_width)
            ymax = min(y_center + h/2, img_height)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            class_ids.append(int(class_id))
    return boxes, class_ids

def boxes_to_masks(boxes, img_height, img_width):
    masks = []
    for box in boxes:
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
       
        xmin, ymin, xmax, ymax = map(int, box)
       
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(img_width, xmax), min(img_height, ymax)

        if xmax > xmin and ymax > ymin:
            mask[ymin:ymax, xmin:xmax] = 1
            masks.append(mask)

    if not masks:
        return np.empty((0, img_height, img_width), dtype=np.uint8)
    return np.array(masks, dtype=np.uint8)

# -----------------------------
# 2. Custom YOLO dataset
# -----------------------------
class SARD2YOLODataset(Dataset):
    
    def __init__(self, images_dir, labels_dir, target_size=640, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_size = target_size 
        self.transforms = transforms
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.txt').replace('.png', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            print(f"Image introuvable ou corrompue : {img_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H_orig, W_orig, _ = img.shape 

        

        # 1. Resize the image
        # Use INTER_AREA for reduction, INTER_LINEAR for enlargement
        interp = cv2.INTER_AREA if H_orig > self.target_size else cv2.INTER_LINEAR
        img_resized = cv2.resize(img, (self.target_size, self.target_size), interpolation=interp)

        # 2. Read the original boxes (based on H_orig, W_orig)
        boxes_orig, class_ids = yolo_to_boxes(label_path, W_orig, H_orig)

        if len(boxes_orig) == 0:
            return None

        # 3. Scale the boxes
        scale_x = self.target_size / W_orig
        scale_y = self.target_size / H_orig

        boxes_scaled = []
        for xmin, ymin, xmax, ymax in boxes_orig:
            boxes_scaled.append([
                xmin * scale_x,
                ymin * scale_y,
                xmax * scale_x,
                ymax * scale_y
            ])

        # 4. Generate masks with the NEW size
        masks_scaled = boxes_to_masks(boxes_scaled, self.target_size, self.target_size)

        

        # Convert to Tensors
        boxes = torch.as_tensor(boxes_scaled, dtype=torch.float32)
        # +1 because 0 is the BACKGROUND class for Mask R-CNN
        labels = torch.as_tensor(class_ids, dtype=torch.int64) + 1
        masks = torch.as_tensor(masks_scaled, dtype=torch.uint8)

        # Handle the case where all boxes are filtered
        if boxes.shape[0] == 0:
            return None

        target = {"boxes": boxes, "labels": labels, "masks": masks}

        # Use the resized image
        img_tensor = F.to_tensor(img_resized)

        return img_tensor, target

# -----------------------------
# 3. Collate_fn for DataLoader
# -----------------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))
