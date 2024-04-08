import cv2
import os
import shutil
import numpy as np
from dataset_preprocess.yolo_detect.dataset_path import (DEFAULT_BEFORE_PATH, hair_images_save_dir, hair_labels_save_dir, hair_plot_save_dir, hair_original_save_dir,
                                                         hair_dca_images_save_dir, hair_dca_labels_save_dir, hair_dca_plot_save_dir, hair_dca_original_save_dir,
                                                         dca_save_dir, dca_plot_save_dir, dca_original_save_dir)
from dataset_preprocess.yolo_dca.mask_creation import YOLODCAMaskCreation
from dataset_preprocess.yolo_hair.mask_creation import YOLOHairMaskCreation
import matplotlib.pyplot as plt

class ImageInfo:
    def __init__(self, image_name, save_dir, labels_dir, move_label):
        self.image_name = image_name
        self.save_dir = save_dir
        self.labels_dir = labels_dir
        self.move_label = move_label
        self.label_path = os.path.join(labels_dir, image_name.replace('.jpg', '.txt'))
        self.plot_save_dir = None
        self.original_save_dir = None  # New attribute for original image save directory

class YOLOMaskedImageGenerator:
    def __init__(self):
        self.hair_mask_creation = YOLOHairMaskCreation()
        self.dca_mask_creation = YOLODCAMaskCreation()

    def detect_dca(self, image):
        hair_exist = self.hair_mask_creation.hair_detected(image)
        dca_exist = self.dca_mask_creation.detect_dark_corners(image)
        return hair_exist, dca_exist

    def combined_mask_creation(self, hair_mask, dca_mask):
        combined_mask = cv2.bitwise_or(hair_mask, dca_mask)
        return combined_mask

    def inpaint_with_combined_mask(self, image, mask, output_path):
        inpainted_image = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)
        cv2.imwrite(output_path, inpainted_image)
    def create_dilated_mask(self, mask, kernel_size = (3, 3), iterations = 1):
        """
        Create a dilated mask from an existing mask to increase the size of the white parts (255) of the mask.
        """
        kernel = np.ones(kernel_size, np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
        return dilated_mask

    def mask_generation(self, images_dir, labels_dir):
        for image_name in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_name)
            image = cv2.imread(image_path)
            hair_exist, dca_exist = self.detect_dca(image)

            hair_mask, dca_mask = None, None

            if hair_exist:
                hair_mask = self.hair_mask_creation.create_hair_mask(image)
                # hair_mask = self.create_dilated_mask(hair_mask)

            if dca_exist:
                dca_mask = self.dca_mask_creation.create_dca_mask(image)
                # dca_mask = self.create_dilated_mask(dca_mask)

            # Determine directories based on detection
            if hair_exist and dca_exist:
                save_dir, label_save_dir, plot_save_dir, original_save_dir = hair_dca_images_save_dir, hair_dca_labels_save_dir, hair_dca_plot_save_dir, hair_dca_original_save_dir
            elif hair_exist:
                save_dir, label_save_dir, plot_save_dir, original_save_dir = hair_images_save_dir, hair_labels_save_dir, hair_plot_save_dir, hair_original_save_dir
            elif dca_exist:
                save_dir, label_save_dir, plot_save_dir, original_save_dir = dca_save_dir, None, dca_plot_save_dir, dca_original_save_dir
            else:
                # If neither hair nor DCA is detected, do not set an original image save directory
                save_dir, label_save_dir, plot_save_dir, original_save_dir = None, None, None, None

            image_info = ImageInfo(image_name, save_dir, labels_dir, bool(label_save_dir))
            image_info.plot_save_dir = plot_save_dir
            image_info.original_save_dir = original_save_dir

            combined_mask = None
            if hair_mask is not None and dca_mask is not None:
                combined_mask = self.combined_mask_creation(hair_mask, dca_mask)
            elif hair_mask is not None:
                combined_mask = hair_mask
            elif dca_mask is not None:
                combined_mask = dca_mask

            if combined_mask is not None:
                self.inpaint_and_save(image, combined_mask, image_info)

            # Save the original image only if a corresponding directory is determined
            if original_save_dir:
                cv2.imwrite(os.path.join(original_save_dir, image_name), image)

    def inpaint_and_save(self, image, mask, image_info: ImageInfo):
        output_path = os.path.join(image_info.save_dir, image_info.image_name)
        self.inpaint_with_combined_mask(image, mask, output_path)
        inpainted_image = cv2.imread(output_path)  # Reload the inpainted image for plot creation

        # Move the label file if necessary
        if image_info.move_label and image_info.labels_dir and os.path.exists(image_info.label_path):
            shutil.move(image_info.label_path,
                        os.path.join(image_info.labels_dir, os.path.basename(image_info.label_path)))

        # Create and save the comparison plot
        plot_output_path = os.path.join(image_info.plot_save_dir, image_info.image_name.replace('.jpg', '.png'))
        self.create_comparison_plot(image, mask, inpainted_image, plot_output_path)

    def create_comparison_plot(self, original_image, mask, inpainted_image, output_path):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title('Mask')
        axs[1].axis('off')

        axs[2].imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
        axs[2].set_title('Inpainted Image')
        axs[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def main():
    yolo_mig = YOLOMaskedImageGenerator()
    os.makedirs(hair_images_save_dir, exist_ok=True)
    os.makedirs(hair_labels_save_dir, exist_ok=True)
    os.makedirs(hair_original_save_dir, exist_ok=True)
    os.makedirs(hair_plot_save_dir, exist_ok=True)
    os.makedirs(hair_dca_images_save_dir, exist_ok=True)
    os.makedirs(hair_dca_labels_save_dir, exist_ok=True)
    os.makedirs(hair_dca_original_save_dir, exist_ok=True)
    os.makedirs(hair_dca_plot_save_dir, exist_ok=True)
    os.makedirs(dca_save_dir, exist_ok=True)
    os.makedirs(dca_original_save_dir, exist_ok=True)
    os.makedirs(dca_plot_save_dir, exist_ok=True)
    for class_dir in os.listdir(DEFAULT_BEFORE_PATH):
        images_dir = os.path.join(DEFAULT_BEFORE_PATH, class_dir, "images")
        labels_dir = os.path.join(DEFAULT_BEFORE_PATH, class_dir, "labels")
        yolo_mig.mask_generation(images_dir, labels_dir)

if __name__ == '__main__':
    main()
