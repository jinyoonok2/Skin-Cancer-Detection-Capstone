import os
import shutil
import re

yolo_dataset_mlc_dir = r"C:\Jinyoon Projects\datasets\combined_dataset_cascade_mela"
yolo_dataset_non_mlc_dir = r"C:\Jinyoon Projects\datasets\combined_dataset_cascade_non_mela"
our_dataset_dir = r"C:\Jinyoon Projects\datasets\combined_dataset_swin_split"

# Define the source directories
yolo_mlc_train_path = os.path.join(yolo_dataset_mlc_dir, 'train', 'images')
yolo_mlc_valid_path = os.path.join(yolo_dataset_mlc_dir, 'valid', 'images')
yolo_mlc_test_path = os.path.join(yolo_dataset_mlc_dir, 'test', 'images')
yolo_non_mlc_train_path = os.path.join(yolo_dataset_non_mlc_dir, 'train', 'images')
yolo_non_mlc_valid_path = os.path.join(yolo_dataset_non_mlc_dir, 'valid', 'images')
yolo_non_mlc_test_path = os.path.join(yolo_dataset_non_mlc_dir, 'test', 'images')

# Define the destination directories
train_mlc_dir = os.path.join(our_dataset_dir, 'train', 'MLC')
valid_mlc_dir = os.path.join(our_dataset_dir, 'valid', 'MLC')
train_non_mlc_dir = os.path.join(our_dataset_dir, 'train', 'NON-MLC')
valid_non_mlc_dir = os.path.join(our_dataset_dir, 'valid', 'NON-MLC')

# Function to create directories if they don't exist
def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# Function to copy files excluding those ending with _number.jpg
def copy_files_excluding_numbered(source, destination):
    for filename in os.listdir(source):
        if re.match(r'.*_[0-9]+\.jpg$', filename):
            continue  # Skip files ending with _number.jpg
        shutil.copy2(os.path.join(source, filename), destination)

# Create the necessary directories
create_dir_if_not_exists(train_mlc_dir)
create_dir_if_not_exists(valid_mlc_dir)
create_dir_if_not_exists(train_non_mlc_dir)
create_dir_if_not_exists(valid_non_mlc_dir)

# Copy the files from source to destination
copy_files_excluding_numbered(yolo_mlc_train_path, train_mlc_dir)
copy_files_excluding_numbered(yolo_mlc_valid_path, train_mlc_dir)
copy_files_excluding_numbered(yolo_mlc_test_path, valid_mlc_dir)
copy_files_excluding_numbered(yolo_non_mlc_train_path, train_non_mlc_dir)
copy_files_excluding_numbered(yolo_non_mlc_valid_path, train_non_mlc_dir)
copy_files_excluding_numbered(yolo_non_mlc_test_path, valid_non_mlc_dir)
