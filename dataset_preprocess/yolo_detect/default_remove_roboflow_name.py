import yaml
import re
import os
from dataset_path import YAML_PATH, DEFAULT_PATH, HAIR_RELABEL_PATH

# Load class names from the YAML file
with open(YAML_PATH, 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']


# Function to rename files in a directory
def rename_files(class_name, path, subdir):
    full_path = os.path.join(path, subdir)

    for filename in os.listdir(full_path):
        # Check if the filename contains the class name
        if class_name in filename:
            # Split the filename at the class name and keep the part before and including the class name
            new_filename = re.split(f'(?<={class_name})', filename)[0] + ('.jpg' if subdir == 'images' else '.txt')
            os.rename(os.path.join(full_path, filename), os.path.join(full_path, new_filename))


# Rename files for DEFAULT_PATH, considering class-specific folders
for class_name in class_names:
    for subdir in ['images', 'labels']:
        # Construct the path for class-specific folders
        path = os.path.join(DEFAULT_PATH, class_name)
        if os.path.exists(path):
            rename_files(class_name, os.path.join(DEFAULT_PATH, class_name), subdir)

# Rename files for HAIR_RELABEL_PATH, where all class files are in the same 'images' or 'labels' directory
for class_name in class_names:
    for subdir in ['images', 'labels']:
        rename_files(class_name, HAIR_RELABEL_PATH, subdir)

print("Renaming process completed.")
