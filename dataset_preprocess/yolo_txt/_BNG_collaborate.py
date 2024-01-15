import shutil
from dataset_path import *
def rename_and_move_files(original_class, new_class):
    for subdir in ['images', 'labels']:
        # Define the paths for the original and new directories
        original_dir = os.path.join(DEFAULT_PATH, original_class, subdir)
        new_dir = os.path.join(DEFAULT_PATH, new_class, subdir)
        os.makedirs(new_dir, exist_ok=True)

        for filename in os.listdir(original_dir):
            # Construct the new filename
            new_filename = filename.replace(original_class, new_class)
            original_file_path = os.path.join(original_dir, filename)
            new_file_path = os.path.join(new_dir, new_filename)

            # Rename and move the file
            shutil.move(original_file_path, new_file_path)
            print(f"Moved {original_file_path} to {new_file_path}")

# List of classes to be combined into the new BNG class
classes_to_combine = ['MNV', 'NV', 'BKL', 'UNK']
new_class_name = 'BNG'

for original_class in classes_to_combine:
    rename_and_move_files(original_class, new_class_name)

print("Files have been successfully renamed and moved.")
