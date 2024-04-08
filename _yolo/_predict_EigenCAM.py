from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
from ultralytics import YOLO
from matplotlib import pyplot as plt
from dataset_preprocess.yolo_detect.dataset_path import (before_model_path, after_model_path, CAM_comparison_before_image_dir,
                                                         CAM_comparison_after_image_dir, CAM_comparison_output_dir)
import cv2
import numpy as np
import os

def process_and_plot_images(before_image_path, after_image_path, output_path, before_model, after_model):
    """
    Process both before and after images to generate and plot their CAM images.
    """
    # Load images
    before_img = cv2.imread(before_image_path)
    after_img = cv2.imread(after_image_path)

    # Process images through models to get CAM images
    before_cam_image = generate_cam_image(before_img, before_model)
    after_cam_image = generate_cam_image(after_img, after_model)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Before Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(before_cam_image)
    axs[0, 1].set_title('Before CAM Image')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('After Image')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(after_cam_image)
    axs[1, 1].set_title('After CAM Image')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_cam_image(image, model):
    """
    Generate CAM image for a single image using a given model.
    """
    # Assuming RGB conversion and normalization steps are done within EigenCAM or are unnecessary
    target_layers = [model.model.model[-2]]  # Example target layer
    cam = EigenCAM(model, target_layers, task='od')
    grayscale_cam = cam(image)[0, :, :]
    cam_image = show_cam_on_image(np.float32(image) / 255, grayscale_cam, use_rgb=True)
    return cam_image


def main():
    # Load models
    before_model = YOLO(before_model_path)
    after_model = YOLO(after_model_path)

    # Iterate through each category to process images
    for category in CAM_comparison_before_image_dir:
        before_dir = CAM_comparison_before_image_dir[category]
        after_dir = CAM_comparison_after_image_dir[category]
        output_dir = CAM_comparison_output_dir[category]
        os.makedirs(output_dir, exist_ok=True)

        for image_name in os.listdir(before_dir):
            before_image_path = os.path.join(before_dir, image_name)
            after_image_path = os.path.join(after_dir, image_name)
            output_path = os.path.join(output_dir, image_name.replace('.jpg', '_comparison.png'))

            process_and_plot_images(before_image_path, after_image_path, output_path, before_model, after_model)


if __name__ == '__main__':
    main()
