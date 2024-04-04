import os
import shutil
import random
import yaml
from dataset_path import *

# Load class names from the YAML file
with open(YAML_PATH, 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['old_names']

# Function to shuffle and split data, then copy to respective directories
def process_class_data(class_name):
    image_path = os.path.join(DEFAULT_PATH, class_name, 'images')
    image_files = os.listdir(image_path)
    random.shuffle(image_files)  # Shuffling for randomness

    # Split ratios
    train_split = int(0.7 * len(image_files))
    valid_split = int(0.9 * len(image_files))

    # Splitting data
    train_files = image_files[:train_split]
    valid_files = image_files[train_split:valid_split]
    test_files = image_files[valid_split:]

    # Function to copy files
    def copy_files(files, data_type):
        for file in files:
            # Copy image
            image_from_class_path = os.path.join(DEFAULT_PATH, f'{class_name}/images/{file}')
            image_to_data_type = os.path.join(DATASET_DIR, f'{data_type}/images/{file}')
            shutil.copy(image_from_class_path, image_to_data_type)

            # Copy corresponding label file
            label_file = file.rsplit('.', 1)[0] + '.txt'
            label_from_class_path = os.path.join(DEFAULT_PATH, f'{class_name}/labels/{label_file}')
            label_to_data_type = os.path.join(DATASET_DIR, f'{data_type}/labels/{label_file}')
            shutil.copy(label_from_class_path, label_to_data_type)

    # Copying files to respective directories
    for data_type, files in zip([TRAIN_PATH, VALID_PATH, TEST_PATH], [train_files, valid_files, test_files]):
        os.makedirs(f'{data_type}/images', exist_ok=True)
        os.makedirs(f'{data_type}/labels', exist_ok=True)
        copy_files(files, data_type)

# Process each class directory
for class_name in class_names:
    process_class_data(class_name)

print("Data separation into train, valid, and test sets completed.")
