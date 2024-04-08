import os

DATASET_DIR = r'C:\Jinyoon_Projects\datasets\hair_dataset' # YOLO dataset
HAIR_PATH = os.path.join(DATASET_DIR, 'HAIR')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

TRAIN_PATH = os.path.join(DATASET_DIR, 'train')
VALID_PATH = os.path.join(DATASET_DIR, 'valid')
TEST_PATH = os.path.join(DATASET_DIR, 'test')

sample_dir = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\dataset_preprocess\yolo_hair\mask_sample"
hair_model_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo_hair\logs\train2-hair\weights\best.pt"