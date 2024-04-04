import os

DATASET_DIR = r'C:\Jinyoon_Projects\datasets\combined_dataset' # YOLO dataset
DEFAULT_PATH = os.path.join(DATASET_DIR, 'default')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
OLD_YAML_PATH = os.path.join(DATASET_DIR, 'data_7.yaml')

TRAIN_PATH = os.path.join(DATASET_DIR, 'train')
VALID_PATH = os.path.join(DATASET_DIR, 'valid')
TEST_PATH = os.path.join(DATASET_DIR, 'test')

extract_dir = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\dataset_preprocess\yolo_dca\mask_sample"
relabel_hair_dir = os.path.join(extract_dir, "relabel_hair")
dca_dir = os.path.join(DATASET_DIR, 'dca')

dca_model_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo_dca\logs\train1-dca\weights\best.pt"
hair_model_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo_hair\logs\train1-hair\weights\best.pt"