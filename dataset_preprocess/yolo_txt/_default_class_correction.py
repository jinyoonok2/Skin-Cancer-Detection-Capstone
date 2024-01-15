import yaml
from dataset_path import *

# Load class names from the YAML file and create a mapping from class name to index
with open(OLD_YAML_PATH, 'r') as file:
    data = yaml.safe_load(file)
    class_name_to_index = {name: index for index, name in enumerate(data['names'])}

# Function to update the class number in label files
def update_label_files(class_name, class_index):
    labels_path = os.path.join(DEFAULT_PATH, class_name, 'labels')
    for label_file in os.listdir(labels_path):
        file_path = os.path.join(labels_path, label_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        with open(file_path, 'w') as file:
            for line in lines:
                parts = line.split(' ')
                # Change the class number to the correct index
                parts[0] = str(class_index)
                # Write the updated line back to the file
                file.write(' '.join(parts))

# Iterate over each class and update label files
for class_name, class_index in class_name_to_index.items():
    update_label_files(class_name, class_index)

print("Label files update completed.")
