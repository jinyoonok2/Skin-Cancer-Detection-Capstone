import os

DATASET_DIR = r'C:\Jinyoon Projects\datasets\combined_dataset_cascade_mela' # YOLO dataset
DEFAULT_PATH = os.path.join(DATASET_DIR, 'default')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

TRAIN_PATH = os.path.join(DATASET_DIR, 'train')
VALID_PATH = os.path.join(DATASET_DIR, 'valid')
TEST_PATH = os.path.join(DATASET_DIR, 'test')