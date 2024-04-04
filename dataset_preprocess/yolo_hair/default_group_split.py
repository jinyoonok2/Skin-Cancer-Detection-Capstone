import os
import shutil
import random
from dataset_preprocess.yolo_hair.dataset_path import DEFAULT_PATH, TRAIN_PATH, VALID_PATH, TEST_PATH

def create_train_valid_test_split(source_dir, seed=42):
    """
    Split images from source_dir into train, valid, and test sets with ratios 7:2:1 and copy them into separate directories.

    :param source_dir: Directory containing class folders with images.
    :param train_dir: Directory for training images.
    :param valid_dir: Directory for validation images.
    :param test_dir: Directory for test images.
    :param seed: Random seed for reproducibility.
    """
    random.seed(seed)  # Ensure reproducibility

    # Iterate through each class folder in the source directory
    for class_folder in os.listdir(source_dir):
        class_source_dir = os.path.join(source_dir, class_folder)

        # Collect all image filenames in the current class directory
        images = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f))]

        # Shuffle the list of images to ensure random selection
        random.shuffle(images)

        # Calculate split sizes
        num_images = len(images)
        num_train = int(0.7 * num_images)
        num_valid = int(0.2 * num_images)
        # The rest goes to the test set

        # Split images based on calculated indices
        train_images = images[:num_train]
        valid_images = images[num_train:num_train + num_valid]
        test_images = images[num_train + num_valid:]

        # Function to copy images to target directory
        def copy_images(images, target_dir):
            os.makedirs(target_dir, exist_ok=True)
            for img in images:
                shutil.copy(os.path.join(class_source_dir, img), os.path.join(target_dir, img))

        # Copy images to their respective directories
        copy_images(train_images, os.path.join(TRAIN_PATH, class_folder))
        copy_images(valid_images, os.path.join(VALID_PATH, class_folder))
        copy_images(test_images, os.path.join(TEST_PATH, class_folder))

if __name__ == '__main__':
    create_train_valid_test_split(DEFAULT_PATH)
