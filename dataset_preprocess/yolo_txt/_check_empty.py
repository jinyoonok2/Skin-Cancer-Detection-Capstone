from dataset_path import *

############################################################################################
# Check Version - train/valid/test
############################################################################################
# Directories to check
directories_to_check = ['train', 'valid', 'test']

# Function to check for empty files in a directory
def check_empty_files(directory):
    labels_path = os.path.join(directory, 'labels')
    if os.path.exists(labels_path):
        for filename in os.listdir(labels_path):
            file_path = os.path.join(labels_path, filename)
            # Check if the file is empty
            if os.path.getsize(file_path) == 0:
                print(f"Empty label file: {file_path}")

# Iterate over train, valid, and test directories and check for empty label files
for directory in directories_to_check:
    os.path.join(DATASET_DIR, directory)
    check_empty_files(directory)

print("Check completed.")

############################################################################################
# Check Version - classes
############################################################################################
# import yaml
# import os
#
# # Load class names from the YAML file
# with open(YAML_PATH, 'r') as file:
#     data = yaml.safe_load(file)
#     class_names = data['names']
#
# # Function to check for empty files in a directory
#
# def check_empty_files(class_name):
#     path = os.path.join(DEFAULT_PATH, class_name, 'labels')
#     for filename in os.listdir(path):
#         file_path = os.path.join(path, filename)
#         # Check if the file is empty
#         if os.path.getsize(file_path) == 0:
#             print(f"Empty label file: {file_path}")
#
# # Iterate over each class and check for empty label files
# for class_name in class_names:
#     check_empty_files(class_name)
#
# print("Check completed.")

############################################################################################
# Move Version
############################################################################################
# import yaml
# import os
# import shutil
#
# # Load class names from the YAML file
# with open(YAML_PATH, 'r') as file:
#     data = yaml.safe_load(file)
#     class_names = data['names']
#
# # Create a directory to store the moved files if it doesn't exist
# moved_files_dir = os.path.join(DATASET_DIR, 'moved_files')
# if not os.path.exists(moved_files_dir):
#     os.makedirs(moved_files_dir)
#
# # Function to move empty label files and corresponding images
# def move_empty_label_files(class_name):
#     labels_path = os.path.join(DEFAULT_PATH, class_name, 'labels')
#     images_path = os.path.join(DEFAULT_PATH, class_name, 'images')
#     for filename in os.listdir(labels_path):
#         file_path = os.path.join(labels_path, filename)
#         # Check if the file is empty
#         if os.path.getsize(file_path) == 0:
#             # Move the label file
#             new_label_path = os.path.join(moved_files_dir, class_name + '_labels', filename)
#             os.makedirs(os.path.dirname(new_label_path), exist_ok=True)
#             shutil.move(file_path, new_label_path)
#             print(f"Moved empty label file: {new_label_path}")
#
#             # Move the corresponding image file
#             image_filename = filename.replace('.txt', '.jpg')
#             image_path = os.path.join(images_path, image_filename)
#             new_image_path = os.path.join(moved_files_dir, class_name + '_images', image_filename)
#             os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
#             if os.path.exists(image_path):
#                 shutil.move(image_path, new_image_path)
#                 print(f"Moved corresponding image file: {new_image_path}")
#
# # Iterate over each class and move empty label files and corresponding images
# for class_name in class_names:
#     move_empty_label_files(class_name)
#
# print("Process completed.")