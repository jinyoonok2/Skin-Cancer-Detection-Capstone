import os
import shutil
import random
from dataset_preprocess.yolo_dca.dataset_path import SKIN_IMAGES_DIR, SKIN_LABELS_DIR, TRAIN_PATH, VALID_PATH

# Assuming these are the class names inferred from your context
class_names = ['MEL', 'MNV', 'NV']

def create_train_valid_split(seed=42):
    random.seed(seed)  # Ensure reproducibility

    # Ensure the structure for train and valid directories exists
    for path in [TRAIN_PATH, VALID_PATH]:
        os.makedirs(os.path.join(path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(path, 'labels'), exist_ok=True)

    # Collect all image and label files
    all_images = os.listdir(SKIN_IMAGES_DIR)
    all_labels = os.listdir(SKIN_LABELS_DIR)

    # Filter files by class and split them
    for class_name in class_names:
        class_images = [img for img in all_images if class_name in img]
        class_labels = [lbl for lbl in all_labels if class_name in lbl.replace('.txt', '.jpg')]

        # Ensure there's a label for each image
        class_images = [img for img in class_images if img.replace('.jpg', '.txt') in class_labels]

        # Shuffle and split the files
        random.shuffle(class_images)
        num_train = int(0.9 * len(class_images))
        # The rest for the validation set

        # Split lists
        train_files = class_images[:num_train]
        valid_files = class_images[num_train:]

        # Copying function
        def copy_files(files, target_dir):
            for file in files:
                src_img_path = os.path.join(SKIN_IMAGES_DIR, file)
                dst_img_path = os.path.join(target_dir, 'images', file)
                src_lbl_path = os.path.join(SKIN_LABELS_DIR, file.replace('.jpg', '.txt'))
                dst_lbl_path = os.path.join(target_dir, 'labels', file.replace('.jpg', '.txt'))

                shutil.copy(src_img_path, dst_img_path)
                shutil.copy(src_lbl_path, dst_lbl_path)

        # Execute copying
        copy_files(train_files, TRAIN_PATH)
        copy_files(valid_files, VALID_PATH)

if __name__ == '__main__':
    create_train_valid_split()
