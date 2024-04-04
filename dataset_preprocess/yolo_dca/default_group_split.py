import os
import shutil
import random
from dataset_preprocess.yolo_dca.dataset_path import SKIN_PATH, TRAIN_PATH, VALID_PATH, TEST_PATH

# Assuming these are the class names inferred from your context
class_names = ['MEL', 'MNV', 'NV']


def create_train_valid_test_split(source_dir, seed=42):
    random.seed(seed)  # Ensure reproducibility

    # Ensure the structure for train, valid, and test directories exists
    for path in [TRAIN_PATH, VALID_PATH, TEST_PATH]:
        os.makedirs(os.path.join(path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(path, 'labels'), exist_ok=True)

    # Collect all image and label files once, since they match by name except extension
    all_images = os.listdir(os.path.join(source_dir, 'images'))
    all_labels = os.listdir(os.path.join(source_dir, 'labels'))

    # Filter files by class and split them
    for class_name in class_names:
        class_images = [img for img in all_images if class_name in img]
        class_labels = [lbl for lbl in all_labels if
                        class_name in lbl.replace('.txt', '.jpg')]  # Assuming label files follow this naming convention

        # Ensure there's a label for each image
        class_images = [img for img in class_images if img.replace('.jpg', '.txt') in class_labels]

        # Shuffle and split the files
        random.shuffle(class_images)
        num_train = int(0.7 * len(class_images))
        num_valid = int(0.2 * len(class_images))
        # The rest for the test set

        # Split lists
        train_files = class_images[:num_train]
        valid_files = class_images[num_train:num_train + num_valid]
        test_files = class_images[num_train + num_valid:]

        # Copying files
        def copy_files(files, target_dir):
            for file in files:
                # Construct source and destination paths for images and labels
                src_img_path = os.path.join(source_dir, 'images', file)
                dst_img_path = os.path.join(target_dir, 'images', file)
                src_lbl_path = os.path.join(source_dir, 'labels', file.replace('.jpg', '.txt'))
                dst_lbl_path = os.path.join(target_dir, 'labels', file.replace('.jpg', '.txt'))

                # Copy files
                shutil.copy(src_img_path, dst_img_path)
                shutil.copy(src_lbl_path, dst_lbl_path)

        # Execute copying
        copy_files(train_files, TRAIN_PATH)
        copy_files(valid_files, VALID_PATH)
        copy_files(test_files, TEST_PATH)


if __name__ == '__main__':
    create_train_valid_test_split(SKIN_PATH)
