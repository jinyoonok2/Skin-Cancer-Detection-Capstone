import os

DATASET_DIR = r'C:\Jinyoon Projects\datasets\combined_dataset' # YOLO dataset
DEFAULT_PATH = os.path.join(DATASET_DIR, 'default')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
OLD_YAML_PATH = os.path.join(DATASET_DIR, 'data_7.yaml')

TRAIN_PATH = os.path.join(DATASET_DIR, 'train')
VALID_PATH = os.path.join(DATASET_DIR, 'valid')
TEST_PATH = os.path.join(DATASET_DIR, 'test')