import yaml
import os
from dataset_path import *

# Load the updated class names and their mapping from the YAML file
with open(YAML_PATH, 'r') as file:
    data = yaml.safe_load(file)
    # Define the classes to be considered as Melanocytic
    melanocytic_classes = ['MNV', 'NV', 'MEL']
    # Initialize the mapping with all classes set to non-melanocytic (classid 1)
    class_name_to_index = {name: 1 for name in data['old_names']}
    # Update the mapping for Melanocytic classes to classid 0
    for name in melanocytic_classes:
        class_name_to_index[name] = 0

# Function to update class numbers in label files
def update_label_files(class_name, class_index):
    labels_path = os.path.join(DEFAULT_PATH, class_name, 'labels')
    if not os.path.exists(labels_path):
        print(f"Path does not exist: {labels_path}")
        return
    for label_file in os.listdir(labels_path):
        file_path = os.path.join(labels_path, label_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        with open(file_path, 'w') as file:
            for line in lines:
                parts = line.split(' ')
                # Update class number to the new index
                parts[0] = str(class_index)
                file.write(' '.join(parts))

# Update label files for each class according to the new class structure
for class_name, class_index in class_name_to_index.items():
    update_label_files(class_name, class_index)

print("Label files update completed.")