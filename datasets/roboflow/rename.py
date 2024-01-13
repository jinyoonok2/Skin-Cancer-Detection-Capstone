import os
import sys
from rename_utils import truncate_after_jpg, capitalize_class_names, process_label_files, load_classes_from_yaml

def process_branch_folder(dataset_folder, branch_folder, classes):
    images_path = os.path.join(dataset_folder, branch_folder, 'images')
    labels_path = os.path.join(dataset_folder, branch_folder, 'labels')

    original_to_new_mapping = truncate_after_jpg(images_path, '.jpg', classes)
    capitalize_class_names(images_path, original_to_new_mapping, '.jpg', classes)
    process_label_files(labels_path, original_to_new_mapping)

def main():
    if len(sys.argv) < 3:
        print("Usage: python rename.py <dataset folder> <branch folder 1> [<branch folder 2> ...]")
        sys.exit(1)

    dataset_folder = sys.argv[1]
    classes = load_classes_from_yaml(dataset_folder)
    branch_folders = sys.argv[2:]

    for branch_folder in branch_folders:
        process_branch_folder(dataset_folder, branch_folder, classes)

    print("Renaming process completed.")

if __name__ == "__main__":
    main()

# Command:
# python rename.py HAM10000v2 train valid test
# ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']