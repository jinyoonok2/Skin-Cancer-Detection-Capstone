import yaml
import os
import re

# Load class names from the YAML file
with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']

# Function to rename files in a directory
def rename_files(class_name, subdir):
    path = os.path.join('./default', class_name, subdir)
    for filename in os.listdir(path):
        # Check if the filename contains the class name
        if class_name in filename:
            # Split the filename at the class name and keep the part before and including the class name
            new_filename = re.split(f'(?<={class_name})', filename)[0] + ('.jpg' if subdir == 'images' else '.txt')
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

# Iterate over each class and rename files in both 'images' and 'labels' subdirectories
for class_name in class_names:
    rename_files(class_name, 'images')
    rename_files(class_name, 'labels')

print("Renaming process completed.")
