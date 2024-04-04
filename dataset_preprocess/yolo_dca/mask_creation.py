import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
from dataset_preprocess.yolo_dca.dataset_path import sample_dir, model_path
from dataset_preprocess.yolo_detect.dataset_path import TEST_PATH

class YOLODCAMaskCreation:
    def __init__(self):
        # Assuming YOLO is a class from a YOLOv5 or similar package that can be initialized with a model path
        self.model = YOLO(model_path)

    def model_predict(self, image):
        # Make prediction on the image
        results = self.model.predict(source=image, imgsz=512, conf=0.35, device=0)
        return results

    def create_dca_mask(self, image):
        original_size = image.shape[:2]
        if original_size != (512, 512):
            image = cv2.resize(image, (512, 512),
                               interpolation=cv2.INTER_AREA if original_size[0] < 512 else cv2.INTER_LINEAR)

        results = self.model_predict(image)
        mask = np.zeros((512, 512), dtype=np.uint8)  # Initialize the mask

        if results and results[0] is not None and results[0].masks is not None:
            segment = results[0].masks.xyn[0]
            points = np.reshape(segment * 512, (-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)  # Fill the desired area with white
            mask = cv2.bitwise_not(mask)
        else:
            print(f"(DCA) No detectable results, skipped.")

        return mask

    def create_segmented_image(self, image):
        original_size = image.shape[:2]
        if original_size != (512, 512):
            image = cv2.resize(image, (512, 512),
                               interpolation=cv2.INTER_AREA if original_size[0] < 512 else cv2.INTER_LINEAR)

        results = self.model_predict(image)

        if results[0] is not None and results[0].masks is not None:
            for segment in results[0].masks.xyn:
                points = np.reshape(segment * 512, (-1, 2)).astype(np.int32)
                cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        else:
            print(f"(DCA) No detectable results, skipped.")

        return image

    def compare_masks_and_save(self, image_path, output_path):
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

    def inpaint_with_combined_mask(self, image, output_path):
        # Step 1: Create the DCA mask
        mask = self.create_dca_mask(image)

        # Step 3: Apply inpainting on the pre-filled image using the original mask
        # Note: Ensure the mask for inpainting is binary (255 for areas to inpaint)
        inpainted_image = cv2.inpaint(image, mask, 10, cv2.INPAINT_TELEA)

        # Step 4: Save the inpainted image
        cv2.imwrite(output_path, inpainted_image)

    def detect_dark_corners(self, image, corner_size=20, darkness_threshold=10, dark_pixel_ratio=0.4):
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
    test_dir = os.path.join(TEST_PATH, 'images')

    # for the comparison purpose
    mask_comparison_dir = os.path.join(sample_dir, "masks_sample")
    os.makedirs(mask_comparison_dir, exist_ok=True)

    # for the inpainting purpose
    inpainting_dir = os.path.join(sample_dir, "inpainting_sample")
    os.makedirs(inpainting_dir, exist_ok=True)

    yolo_dca_interface = YOLODCAMaskCreation()

    for image_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_name)
        mask_comparison_path = os.path.join(mask_comparison_dir, image_name)
        inpainted_image_path = os.path.join(inpainting_dir, image_name)

        # Load the image
        image = cv2.imread(image_path)

        # Check for dark corners before processing
        if yolo_dca_interface.detect_dark_corners(image):
            # Process only if dark corners are detected
            yolo_dca_interface.compare_masks_and_save(image, mask_comparison_path)
            yolo_dca_interface.inpaint_with_combined_mask(image, inpainted_image_path)
        else:
            print(f"No dark corners detected in {image_name}, skipping...")

if __name__ == '__main__':
    main()