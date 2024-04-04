from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os

def process_image(image_path, output_path):
    # Assuming model is loaded outside this function for efficiency
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    rgb_img = img.copy()
    img = np.float32(img) / 255

    # Access Target Layers
    target_layers = [model.model.model[-4]]

    # Utilize EigenCAM & prepare images for plotting
    cam = EigenCAM(model, target_layers, task='od')
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    # Prepare grayscale image for plotting
    g_scale = np.stack([grayscale_cam] * 3, axis=2)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(g_scale, cmap='gray')
    axs[1].set_title('Grayscale CAM')
    axs[1].axis('off')

    axs[2].imshow(cam_image)
    axs[2].set_title('CAM Image')
    axs[2].axis('off')

    # Save figure to output_path
    plt.savefig(os.path.join(output_path, os.path.basename(image_path)))
    plt.close(fig)  # Close the figure to free memory

if __name__ == '__main__':
    model_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo\logs\detect\train1-class3-small\weights\best.pt"
    model = YOLO(model_path)  # Load model only once

    directory_path = r"C:\Jinyoon_Projects\datasets\combined_dataset\archieve\class_3\deafulat_512\MEL\images"
    output_directory_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo\cam_sample"
    os.makedirs(output_directory_path, exist_ok=True)

    for filename in os.listdir(directory_path):
        if filename.endswith((".jpg", ".png")):  # Add any other file extensions as needed
            image_path = os.path.join(directory_path, filename)
            process_image(image_path, output_directory_path)