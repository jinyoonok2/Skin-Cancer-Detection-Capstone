import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
from dataset_preprocess.yolo_dca.dataset_path import dca_seg_model_path, sample_dir
from dataset_preprocess.yolo_detect.dataset_path import TEST_PATH
import shutil

class YOLODCAMaskCreation:
    def __init__(self):
        # Assuming YOLO is a class from a YOLOv5 or similar package that can be initialized with a model path
        self.model = YOLO(dca_seg_model_path)

    def _create_adjusted_threshold_mask(self, shape, edge_threshold):
        """
        Create a mask where the threshold for detecting dark regions starts at 0 in the center
        and linearly increases to `edge_threshold` at the edges.
        """
        rows, cols = shape[:2]
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        X, Y = np.meshgrid(x, y)
        # Calculate distance to center normalized to [0,1]
        D = np.sqrt(X ** 2 + Y ** 2) / np.sqrt(2)
        # Linearly scale distance so threshold at center is 0 and `edge_threshold` at edges
        threshold_mask = np.interp(D, [0, 1], [0, edge_threshold])
        return threshold_mask

    def _create_dca_mask_black_threshold(self, image, edge_darkness_threshold=15, kernel_size=(5, 5), iterations=1):
        """
        Create a binary mask for the purely black parts of an image, with no dark regions detected
        in the center and a linear gradient to `edge_darkness_threshold` at the boundaries.
        This function now also applies dilation to the mask to increase the size of the white parts.
        """
        grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        threshold_mask = self.create_adjusted_threshold_mask(image.shape, edge_darkness_threshold)
        mask = np.where(grayScale <= threshold_mask, 255, 0).astype(np.uint8)

        # Apply dilation to the mask before returning
        dilated_mask = self.create_dilated_mask(mask, kernel_size=kernel_size, iterations=iterations)
        return dilated_mask

    def _compare_masks_and_save(self, image, output_path):
        # Load the original image
        # Ensure the image is in RGB format for displaying
        original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate the mask with the new function
        mask = self.create_dca_mask_black_threshold(image)

        # Create a copy of the original image to apply the mask
        image_with_mask = original_image_rgb.copy()
        # Apply the red color to the 255 regions of the mask
        image_with_mask[np.where(mask == 255)] = [255, 0, 0]  # Red in RGB

        # Create a figure with a specific size
        plt.figure(figsize=(10, 5))

        # Plot the original image
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.imshow(original_image_rgb)  # Display the original image in RGB
        plt.title('Original Image')
        plt.axis('off')  # Hide the axes ticks

        # Plot the image applied with the mask
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.imshow(image_with_mask)  # Display the image with the mask applied
        plt.title('Image with Mask Applied')
        plt.axis('off')  # Hide the axes ticks

        # Save the figure to the specified output path
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

    def model_predict(self, image):
        # Make prediction on the image
        results = self.model.predict(source=image, imgsz=512, conf=0.25, device=0)
        return results

    def create_dca_mask(self, image):
        original_size = image.shape[:2]
        if original_size != (512, 512):
            image = cv2.resize(image, (512, 512),
                               interpolation=cv2.INTER_AREA if original_size[0] < 512 else cv2.INTER_LINEAR)

        results = self.model_predict(image)
        combined_mask = np.zeros((512, 512), dtype=np.uint8)  # Initialize the combined mask
        detection_found = False  # Track if any detections have occurred

        for result in results:
            if result is not None and result.masks is not None:
                detection_found = True  # Update flag to indicate detection
                for segment in result.masks.xyn:
                    mask = np.zeros((512, 512), dtype=np.uint8)  # Initialize the mask for the current object
                    points = np.reshape(segment * 512, (-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [points], 255)  # Fill the desired area with white
                    combined_mask = cv2.bitwise_or(combined_mask,
                                                   mask)  # Combine the current mask with the combined mask

        if detection_found:
            combined_mask = cv2.bitwise_not(combined_mask)  # Invert the combined mask only if a detection was found

        if not detection_found:  # Check if no detections were found
            print(f"(DCA) No detectable results, return empty mask.")

        return combined_mask

    def create_segmented_image(self, image):
        original_size = image.shape[:2]
        if original_size != (512, 512):
            image = cv2.resize(image, (512, 512),
                               interpolation=cv2.INTER_AREA if original_size[0] < 512 else cv2.INTER_LINEAR)

        results = self.model_predict(image)

        for result in results:
            if result is not None and result.masks is not None:
                for segment in result.masks.xyn:
                    points = np.reshape(segment * 512, (-1, 2)).astype(np.int32)
                    cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

        if not results or all(result is None or result.masks is None for result in results):
            print(f"(DCA) No detectable results, return empty mask.")

        return image

    def compare_masks_and_save_ml(self, image_path, output_path):
        # Generate the DCA mask and segmented image as before
        mask = self.create_dca_mask(image_path)
        segmented_image = self.create_segmented_image(image_path)

        # Convert mask to binary format (0 and 1) for display
        mask_for_display = mask // 255  # Ensures mask is in binary form

        # Create a figure with a specific size
        plt.figure(figsize=(10, 5))

        # Plot the binary mask
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.imshow(mask_for_display, cmap='gray', vmin=0, vmax=1)  # Direct binary display
        plt.title('DCA Mask')
        plt.axis('off')  # Hide the axes ticks

        # Plot the segmented image
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
        plt.title('Segmented Image')
        plt.axis('off')  # Hide the axes ticks

        # Save the figure to the specified output path
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

    def detect_dark_corners(self, image, corner_size=10, darkness_threshold=10, dark_pixel_ratio=0.2):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners = {
            'top_left': gray[:corner_size, :corner_size],
            'top_right': gray[:corner_size, -corner_size:],
            'bottom_left': gray[-corner_size:, :corner_size],
            'bottom_right': gray[-corner_size:, -corner_size:]
        }

        dark_corners_detected = 0
        for _, corner in corners.items():
            dark_pixels = np.sum(corner < darkness_threshold)
            if dark_pixels / (corner_size * corner_size) > dark_pixel_ratio:
                dark_corners_detected += 1

        return dark_corners_detected > 0

def main():
    image_dir = r"C:\Jinyoon_Projects\datasets\combined_dataset\confounding_removal_before\seed_42\train\images"
    os.makedirs(sample_dir, exist_ok=True)

    yolo_dca_interface = YOLODCAMaskCreation()

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        mask_comparison_path = os.path.join(sample_dir, image_name)
        # inpainted_image_path = os.path.join(inpainting_dir, image_name)

        # Load the image
        image = cv2.imread(image_path)

        # Check for dark corners before processing
        if yolo_dca_interface.detect_dark_corners(image):
            # Process only if dark corners are detected
            yolo_dca_interface.compare_masks_and_save_ml(image, mask_comparison_path)
        else:
            print(f"No dark corners detected in {image_name}, skipping...")


if __name__ == '__main__':
    main()