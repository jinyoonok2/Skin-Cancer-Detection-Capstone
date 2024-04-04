import os
import shutil
from coco_dataset_path import *

# Define magnification factors for each class
magnification_factors = {
    'MEL': 8, 'BCC': 8, 'AK': 16, 'DF': 16, 'VASC': 16, 'SCC': 24, 'UNK': 1
}

def delete_duplicated_files(class_name, filename, file_type):
    magnification = magnification_factors[class_name]
    for i in range(2, magnification + 1):  # Files are numbered from 2 to magnification
        duplicated_filename = filename.rsplit('.', 1)[0] + f'_{i}.' + filename.rsplit('.', 1)[1]
        file_to_delete = os.path.join(COCO_TRAIN_PATH, file_type, duplicated_filename)
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)

def main():
    # Process each image and label file in the train folder
    for file_type in ['images', 'labels']:
        path = os.path.join(COCO_TRAIN_PATH, file_type)
        for filename in os.listdir(path):
            # Extract class name from filename
            class_name = filename.rsplit('_', 1)[-1].rsplit('.', 1)[0]
            if class_name in magnification_factors:
                delete_duplicated_files(class_name, filename, file_type)

    print("Deletion of duplicated files based on class-specific magnification completed.")

if __name__ == "__main__":
    main()
