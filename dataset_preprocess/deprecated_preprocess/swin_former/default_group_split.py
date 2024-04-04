import os
import shutil
import yaml
from dataset_preprocess.swin_former.dataset_path import DEFAULT_PATH, YAML_PATH

# Function to read class names and mappings from YAML
def read_class_mappings(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['old_names'], data['names']

# Function to create directories
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Function to process and distribute files
def distribute_files(dataset_path, train_path, valid_path, old_names, new_classes):
    mlc_classes = ['MEL', 'MNV', 'NV']
    for old_name in old_names:
        new_class = 'MLC' if old_name in mlc_classes else 'NON-MLC'
        source_dir = os.path.join(dataset_path, old_name)
        files = os.listdir(source_dir)
        split_index = int(len(files) * 0.9)
        # Train files
        train_files = files[:split_index]
        # Validation files
        valid_files = files[split_index:]
        # Copy files to new directories
        for file in train_files:
            dest_dir = os.path.join(train_path, new_class)
            ensure_dir(dest_dir)
            shutil.copy(os.path.join(source_dir, file), dest_dir)
        for file in valid_files:
            dest_dir = os.path.join(valid_path, new_class)
            ensure_dir(dest_dir)
            shutil.copy(os.path.join(source_dir, file), dest_dir)

# Main script
if __name__ == "__main__":
    TRAIN_PATH = 'train'
    VALID_PATH = 'valid'
    old_names, new_classes = read_class_mappings(YAML_PATH)
    distribute_files(DEFAULT_PATH, TRAIN_PATH, VALID_PATH, old_names, new_classes)