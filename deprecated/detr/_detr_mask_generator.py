import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from dataset_preprocess.detr_coco_json.coco_dataset_path import *

def create_mask_from_polygon(image_shape, polygon):
    # Reshape flat list into a 2D array [(x1, y1), (x2, y2), ...]
    polygon = np.array(polygon).reshape(-1, 2)
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask


def create_masks_from_coco_json(json_file, output_dir):
    with open(json_file, 'r') as file:
        coco_data = json.load(file)

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']

        # Find corresponding image info
        image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
        if not image_info:
            continue

        image_shape = (image_info['height'], image_info['width'])
        mask = np.zeros(image_shape, dtype=np.uint8)

        for segment in segmentation:
            mask |= create_mask_from_polygon(image_shape, segment)

        mask_image = Image.fromarray(mask * 255)
        mask_filename = f"{image_info['file_name'].split('.')[0]}_mask_{category_id}.png"
        mask_image.save(Path(output_dir) / mask_filename)

# Example usage
if __name__ == '__main__':
    # mask_path = os.path.join(COCO_TRAIN_PATH, 'masks')
    # os.makedirs(mask_path, exist_ok=True)
    # create_masks_from_coco_json(COCO_TRAIN_ANNOTATION_PATH, mask_path)

    # mask_path = os.path.join(COCO_VALID_PATH, 'masks')
    # os.makedirs(mask_path, exist_ok=True)
    # create_masks_from_coco_json(COCO_VALID_ANNOTATION_PATH, mask_path)

    mask_path = os.path.join(COCO_TEST_PATH, 'masks')
    os.makedirs(mask_path, exist_ok=True)
    create_masks_from_coco_json(COCO_TEST_ANNOTATION_PATH, mask_path)
