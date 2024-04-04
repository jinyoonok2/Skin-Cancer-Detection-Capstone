import re
import os
from dataset_preprocess.yolo_dca.dataset_path import SKIN_PATH

# Load class names from the YAML file
class_names = ['MEL', 'MNV', 'NV']

# Function to rename files in a directory
def rename_files(subdir):
    path = os.path.join(SKIN_PATH, subdir)
    for filename in os.listdir(path):
        # Check if the filename contains any of the class names
        if any(class_name in filename for class_name in class_names):
            for class_name in class_names:
                if class_name in filename:
                    # Use regex to find the pattern up to and including the class name
                    pattern = re.compile(f'.*{class_name}')
                    match = pattern.match(filename)
                    if match:
                        new_filename = match.group() + ('.jpg' if subdir == 'images' else '.txt')
                        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
                    break  # Found the class name in the filename, no need to continue checking

rename_files('images')
rename_files('labels')

print("Renaming process completed.")
