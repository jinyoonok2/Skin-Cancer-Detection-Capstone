import cv2
import os
from dataset_preprocess.yolo_detect.dataset_path import relabel_hair_dir, dca_dir, hair_model_path, dca_model_path, TRAIN_PATH, VALID_PATH, TEST_PATH
from dataset_preprocess.yolo_dca.mask_creation import YOLODCAMaskCreation
from dataset_preprocess.yolo_hair.mask_creation import YOLOHairMaskCreation
import numpy as np
from ultralytics import YOLO
import torch
class YOLOExtractRelabel:
    def __init__(self):
        self.hair_model = YOLO(hair_model_path)
        self.dca_model = YOLO(dca_model_path)
        self.hair_mask_creation = YOLOHairMaskCreation()
        self.dca_mask_creation = YOLODCAMaskCreation()

def main():
    return

if __name__ == '__main__':
    main()