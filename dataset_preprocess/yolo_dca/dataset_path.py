import os

DATASET_DIR = r'C:\Jinyoon_Projects\datasets\dca_dataset' # YOLO dataset
SKIN_PATH = os.path.join(DATASET_DIR, 'SKIN')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

TRAIN_PATH = os.path.join(DATASET_DIR, 'train')
VALID_PATH = os.path.join(DATASET_DIR, 'valid')
TEST_PATH = os.path.join(DATASET_DIR, 'test')

sample_dir = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\dataset_preprocess\yolo_dca\mask_sample"
model_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo_dca\logs\train1-dca\weights\best.pt"

class_names = ["MEL", "MNV", "NV"]  # You can modify this list as needed