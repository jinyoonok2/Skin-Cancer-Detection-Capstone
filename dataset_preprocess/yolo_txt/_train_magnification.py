import shutil
from dataset_path import *

# Define magnification factors for each class
# magnification_factors = {
#     'MEL': 8, 'BCC': 8, 'AK': 16, 'DF': 16, 'VASC': 16, 'SCC': 24, 'BNG': 1
# }

magnification_factors = {
    'MEL': 4, 'MNV': 2, 'NV': 2, 'BCC': 4, 'AK': 16,
    'BKL': 8, 'DF': 16, 'VASC': 8, 'SCC': 16, 'UNK': 1
}

def duplicate_files(class_name, filename, file_type):
    magnification = magnification_factors[class_name]
    for i in range(2, magnification + 1):  # Start numbering from 2
        new_filename = filename.rsplit('.', 1)[0] + f'_{i}.' + filename.rsplit('.', 1)[1]
        old_name = os.path.join(TRAIN_PATH, file_type, filename)
        new_name = os.path.join(TRAIN_PATH, file_type, new_filename)
        shutil.copy(old_name, new_name)

def main():
    # Process each image and label file in the train folder
    for file_type in ['images', 'labels']:
        path = os.path.join(TRAIN_PATH, file_type)
        for filename in os.listdir(path):
            # Extract class name from filename
            class_name = filename.rsplit('_', 1)[-1].rsplit('.', 1)[0]
            if class_name in magnification_factors:
                duplicate_files(class_name, filename, file_type)

    print("File duplication based on class-specific magnification completed.")

if __name__ == "__main__":
    main()


