import os
import cv2
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# Define paths
from dataset_path import YAML_PATH, TRAIN_PATH_IMAGES, TRAIN_PATH_LABELS

# Create 'conversion_sample' directory if it doesn't exist
output_dir = 'conversion_sample'
os.makedirs(output_dir, exist_ok=True)

# Load class labels
with open(YAML_PATH, 'r') as yamlfile:
    cfg = yaml.safe_load(yamlfile)
id2label = {idx: label for idx, label in enumerate(cfg['names'])}
print("id2label:", id2label)

# Function to create masks
def create_masks(label_file, img_shape):
    red_channel = np.zeros(img_shape, dtype=np.uint8)
    green_channel = np.zeros(img_shape, dtype=np.uint8)
    blue_channel = np.zeros(img_shape, dtype=np.uint8)  # Empty channel

    with open(label_file, 'r') as file:
        lines = file.readlines()
        instance_id = 1
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            points = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
            points[:, 0] *= img_shape[1]  # Denormalize x
            points[:, 1] *= img_shape[0]  # Denormalize y
            points = points.astype(np.int32)

            # Draw polygon on green channel with the instance ID
            cv2.fillPoly(green_channel, [points], color=instance_id)

            # Directly use class_id without offsetting or scaling
            cv2.fillPoly(red_channel, [points], color=class_id)

            instance_id += 1

    # Merge channels to create full mask
    full_mask = cv2.merge([blue_channel, green_channel, red_channel])
    return full_mask, red_channel, green_channel

# Function to visualize masks
def visualize_masks(image, red_mask, green_mask, full_mask, sample_idx):
    # Original Image
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title(f'Original Image {sample_idx}')
    plt.axis('off')

    # Red Channel Mask
    plt.subplot(2, 2, 2)
    plt.imshow(red_mask, cmap='gray')
    plt.title(f'Red Channel Mask (Class) {sample_idx}')
    plt.axis('off')

    # Green Channel Mask
    plt.subplot(2, 2, 3)
    plt.imshow(green_mask, cmap='gray')
    plt.title(f'Green Channel Mask (Instance) {sample_idx}')
    plt.axis('off')

    # Full Mask
    plt.subplot(2, 2, 4)
    plt.imshow(full_mask * 255)  # Scale up to use the full [0, 255] range
    plt.title(f'Full Mask {sample_idx}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Process one sample and visualize
def process_sample_and_visualize(image_path, label_path, sample_idx):
    # Read image and get shape
    image = cv2.imread(str(image_path))
    img_shape = image.shape[:2]

    # Create masks
    full_mask, red_mask, green_mask = create_masks(label_path, img_shape)

    # Visualize masks
    visualize_masks(image, red_mask, green_mask, full_mask, sample_idx)

# Process and visualize the first 20 samples
for idx, sample_label_file in enumerate(Path(TRAIN_PATH_LABELS).glob('*.txt')):
    if idx >= 3:  # Limit to 20 samples
        break
    sample_image_file = Path(TRAIN_PATH_IMAGES) / (sample_label_file.stem + '.jpg')
    process_sample_and_visualize(sample_image_file, sample_label_file, idx + 1)
