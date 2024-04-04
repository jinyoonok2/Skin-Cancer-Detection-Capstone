import os
import yaml

DATASET_DIR = r"C:\Jinyoon Projects\datasets\combined_dataset_semantic"

YAML_PATH = os.path.join(DATASET_DIR, "COMB_DATA_YAML.yaml")

DEFAULT_PATH = os.path.join(DATASET_DIR, "default")

COMB_TRAIN_PATH = os.path.join(DATASET_DIR, "train_valid_combined")
TRAIN_PATH = os.path.join(DATASET_DIR, "train")
VALID_PATH = os.path.join(DATASET_DIR, "valid")
TEST_PATH = os.path.join(DATASET_DIR, "test")

COMB_TRAIN_PATH_IMAGES = os.path.join(COMB_TRAIN_PATH, "images")
TRAIN_PATH_IMAGES = os.path.join(TRAIN_PATH, "images")
VALID_PATH_IMAGES = os.path.join(VALID_PATH, "images")
TEST_PATH_IMAGES = os.path.join(TEST_PATH, "images")

COMB_TRAIN_PATH_LABELS = os.path.join(COMB_TRAIN_PATH, "labels")
TRAIN_PATH_LABELS = os.path.join(TRAIN_PATH, "labels")
VALID_PATH_LABELS = os.path.join(VALID_PATH, "labels")
TEST_PATH_LABELS = os.path.join(TEST_PATH, "label")

COMB_TRAIN_PATH_MASKS = os.path.join(COMB_TRAIN_PATH, "masks")
TRAIN_PATH_MASKS = os.path.join(TRAIN_PATH, "masks")
VALID_PATH_MASKS = os.path.join(VALID_PATH, "masks")
TEST_PATH_MASKS = os.path.join(TEST_PATH, "masks")

# Load class names from a YAML file
with open(YAML_PATH, 'r') as file:
    class_names = yaml.safe_load(file)
# Assuming the class names are under the 'names' key in your YAML file
CLASSES = class_names['names']
# Create id2label dictionary
ID2LABEL = {idx: label for idx, label in enumerate(CLASSES)}