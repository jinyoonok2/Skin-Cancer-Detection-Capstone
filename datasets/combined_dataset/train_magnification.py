import os
import shutil

# Define magnification factors for each class
magnification_factors = {
    'MEL': 4, 'MNV': 2, 'NV': 2, 'BCC': 4, 'AK': 16,
    'BKL': 8, 'DF': 16, 'VASC': 8, 'SCC': 16, 'UNK': 1
}

# MEL: 0.531
# MNV: 0.38
# NV: 0.892
# BCC: 0.532
# AK: 0.29
# BKL: 0.189
# DF: 0.409
# VASC: 0.804
# SCC: 0.214
# UNK: 0.948

def duplicate_files(class_name, filename, file_type):
    magnification = magnification_factors[class_name]
    for i in range(2, magnification + 1):  # Start numbering from 2
        new_filename = filename.rsplit('.', 1)[0] + f'_{i}.' + filename.rsplit('.', 1)[1]
        shutil.copy(f'./train/{file_type}/{filename}', f'./train/{file_type}/{new_filename}')

if __name__ == "__main__":
    # Process each image and label file in the train folder
    for file_type in ['images', 'labels']:
        path = f'./train/{file_type}'
        for filename in os.listdir(path):
            # Extract class name from filename
            class_name = filename.rsplit('_', 1)[-1].rsplit('.', 1)[0]
            if class_name in magnification_factors:
                duplicate_files(class_name, filename, file_type)

    print("File duplication based on class-specific magnification completed.")

