import cv2
import os
from dataset_preprocess.yolo_hair.dataset_path import sample_dir, hair_model_path
from dataset_preprocess.yolo_detect.dataset_path import TEST_PATH
import numpy as np
from ultralytics import YOLO
import torch
class YOLOHairMaskCreation:
    def __init__(self):
        self.model = YOLO(hair_model_path)  # Initialize YOLO with the provided model path

    def hair_detected(self, image):
        results = self.model.predict(source=image, imgsz=512, device=0)
        top1 = results[0].probs.top1
        return top1 == 0  # Use torch's argmax

    def create_hair_mask(self, image, darkness_threshold=10):
        grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(1, (9, 9))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, darkness_threshold, 255, cv2.THRESH_BINARY)
        return mask

    def save_mask(self, mask, output_path):
        cv2.imwrite(output_path, mask)

def main():
    yolo_hair_mask_creator = YOLOHairMaskCreation()
    test_images_dir = os.path.join(TEST_PATH, 'images')  # Ensure this is the correct path
    os.makedirs(sample_dir, exist_ok=True)

    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping {image_name}, unable to load.")
            continue

        if yolo_hair_mask_creator.hair_detected(image):
            mask = yolo_hair_mask_creator.create_hair_mask(image)
        else:
            # Create an empty mask for images without hair detected
            mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

        output_path = os.path.join(sample_dir, image_name)  # Adjust this path as needed
        yolo_hair_mask_creator.save_mask(mask, output_path)

if __name__ == '__main__':
    main()