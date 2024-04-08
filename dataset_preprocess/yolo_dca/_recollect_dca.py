from dataset_preprocess.yolo_dca.mask_creation import YOLODCAMaskCreation
from dataset_preprocess.yolo_dca.dataset_path import (missed_images_dir, missed_labels_dir, SKIN_IMAGES_DIR, model_path)
from dataset_preprocess.yolo_detect.dataset_path import DEFAULT_BEFORE_PATH
from ultralytics import YOLO
import os
import shutil
import cv2


def process_images_for_dark_corners(default_before_path, skin_images_dir, missed_images_dir):
    # Initialize the YOLO DCA mask creation object
    yolo_dca_mc = YOLODCAMaskCreation()

    # Iterate through each subdirectory in the DEFAULT_BEFORE_PATH
    for subdir in os.listdir(default_before_path):
        images_dir = os.path.join(default_before_path, subdir, "images")
        if os.path.isdir(images_dir):
            # Iterate through each image in the "images" folder
            for image_name in os.listdir(images_dir):
                image_path = os.path.join(images_dir, image_name)

                # Load the image with cv2
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image {image_name}. Skipping...")
                    continue

                # Check for dark corners
                image_has_dark_corners = yolo_dca_mc.detect_dark_corners(image)

                # Check for dark corners and corresponding image in SKIN_IMAGES_DIR
                if image_has_dark_corners:
                    corresponding_skin_image_path = os.path.join(skin_images_dir, image_name)
                    if not os.path.exists(corresponding_skin_image_path):
                        # Copy the image to missed_image_dir if no corresponding image is found
                        shutil.copy(image_path, os.path.join(missed_images_dir, image_name))
                        print(f"Image {image_name} copied to {missed_images_dir}.")


if __name__ == '__main__':
    os.makedirs(missed_images_dir, exist_ok=True)
    os.makedirs(missed_labels_dir, exist_ok=True)
    process_images_for_dark_corners(DEFAULT_BEFORE_PATH, SKIN_IMAGES_DIR, missed_images_dir)


# DEFAULT_BEFORE_PATH has folders that each contains "images" folders and each images folder contains image files.
# as you go through images, check image with detect_dark_corners like the following. if it returns true, then it has dark corner.
#
# yolo_dca_mc = YOLODCAMaskCreation()
# yolo_dca_mc.detect_dark_corners(image)
#
# and if it has dark corner, then check if this is in SKIN_IMAGES_DIR. if there is corresponding image with the same name, move on
# but if there is no corresponding image with same name, copy the image to missed_image_dir.
# can you create this code and show it to me?