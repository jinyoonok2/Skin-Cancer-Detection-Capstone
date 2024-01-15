import os
import shutil
import random
import yaml

# Load class names from the YAML file
with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']

# Function to shuffle and split data, then copy to respective directories
def process_class_data(class_name):
    image_files = os.listdir(f'./default/{class_name}/images')
    random.shuffle(image_files)  # Shuffling for randomness

    # Split ratios
    train_split = int(0.7 * len(image_files))
    valid_split = int(0.9 * len(image_files))

    # Splitting data
    train_files = image_files[:train_split]
    valid_files = image_files[train_split:valid_split]
    test_files = image_files[valid_split:]

    # Function to copy files
    def copy_files(files, data_type):
        for file in files:
            # Copy image
            shutil.copy(f'./default/{class_name}/images/{file}', f'./{data_type}/images/{file}')
            # Copy corresponding label file
            label_file = file.rsplit('.', 1)[0] + '.txt'
            shutil.copy(f'./default/{class_name}/labels/{label_file}', f'./{data_type}/labels/{label_file}')

    # Copying files to respective directories
    for data_type, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        os.makedirs(f'./{data_type}/images', exist_ok=True)
        os.makedirs(f'./{data_type}/labels', exist_ok=True)
        copy_files(files, data_type)

# Process each class directory
for class_name in class_names:
    process_class_data(class_name)

print("Data separation into train, valid, and test sets completed.")
