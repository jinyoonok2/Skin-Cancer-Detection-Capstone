# import os
# import cv2
# import numpy as np
# from pathlib import Path
#
# # # Define paths
# # from dataset_path import TRAIN_PATH_MASKS, TRAIN_PATH_LABELS, VALID_PATH_MASKS, VALID_PATH_LABELS, TEST_PATH_MASKS, TEST_PATH_LABELS, COMB_TRAIN_PATH_LABELS, COMB_TRAIN_PATH_MASKS
# # Define paths
# from dataset_path import TEST_PATH_MASKS, TEST_PATH_LABELS, COMB_TRAIN_PATH_LABELS, COMB_TRAIN_PATH_MASKS
#
# # Function to create masks
# def create_masks(label_file, img_shape):
#     red_channel = np.zeros(img_shape, dtype=np.uint8)
#     green_channel = np.zeros(img_shape, dtype=np.uint8)
#     blue_channel = np.zeros(img_shape, dtype=np.uint8)  # Empty channel
#
#     with open(label_file, 'r') as file:
#         lines = file.readlines()
#         instance_id = 1
#         for line in lines:
#             parts = line.strip().split()
#             class_id = int(parts[0])
#             points = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
#             points[:, 0] *= img_shape[1]  # Denormalize x
#             points[:, 1] *= img_shape[0]  # Denormalize y
#             points = points.astype(np.int32)
#
#             # Draw polygon on green channel with the instance ID
#             cv2.fillPoly(green_channel, [points], color=instance_id)
#
#             # Increment class_id by 1 and use it for the red channel
#             # This will distinguish class IDs from the background (0)
#             cv2.fillPoly(red_channel, [points], color=class_id + 1)
#
#             instance_id += 1
#
#     # Merge channels to create full mask
#     full_mask = cv2.merge([blue_channel, green_channel, red_channel])
#     return full_mask
#
# # Function to save mask debugging information to a file
# # def save_debug_info(debug_file_path, label_file, red_channel):
# #     unique_values, counts = np.unique(red_channel, return_counts=True)
# #     with open(debug_file_path, 'a') as debug_file:  # Append mode
# #         debug_file.write(f"Processing file: {label_file}\n")
# #         debug_file.write(f"Class IDs and counts in the mask: {dict(zip(unique_values, counts))}\n")
# #         debug_file.write("\n")  # Add an extra newline for better readability
#
# # Function to process the label files and create masks
# def process_labels(label_path, mask_path, debug_file_path):
#     os.makedirs(mask_path, exist_ok=True)  # Create mask path if it doesn't exist
#     for label_file in Path(label_path).glob('*.txt'):
#         # Read label file and create mask
#         img_shape = (256, 256)  # Fixed image size
#         full_mask = create_masks(label_file, img_shape)
#
#         # # Save debugging information about the mask creation
#         # save_debug_info(debug_file_path, label_file, full_mask[:, :, 2])  # Red channel contains class_id information
#
#         # Save full mask to mask path
#         mask_filename = os.path.join(mask_path, label_file.stem + '_mask.png')
#         cv2.imwrite(mask_filename, full_mask)
#
# # Define the path for the debug file
# debug_file_path = 'mask_debug_info.txt'  # Adjust the path as needed
#
# # Process label files for training, validation, and testing
# process_labels(COMB_TRAIN_PATH_LABELS, COMB_TRAIN_PATH_MASKS, debug_file_path)
# # process_labels(TRAIN_PATH_LABELS, TRAIN_PATH_MASKS, debug_file_path)
# # process_labels(VALID_PATH_LABELS, VALID_PATH_MASKS, debug_file_path)
# process_labels(TEST_PATH_LABELS, TEST_PATH_MASKS, debug_file_path)

