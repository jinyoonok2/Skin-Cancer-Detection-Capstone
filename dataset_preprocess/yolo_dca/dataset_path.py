import os

DATASET_DIR = r'C:\Jinyoon_Projects\datasets\dca_dataset' # YOLO dataset
# Skin images and labels dataset to separate dca
SKIN_PATH = os.path.join(DATASET_DIR, 'SKIN')
SKIN_IMAGES_DIR = os.path.join(SKIN_PATH, 'images')
SKIN_LABELS_DIR = os.path.join(SKIN_PATH, 'labels')

YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

TRAIN_PATH = os.path.join(DATASET_DIR, 'train')
VALID_PATH = os.path.join(DATASET_DIR, 'valid')
TEST_PATH = os.path.join(DATASET_DIR, 'test')

# trained model on dca
dca_seg_model_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo_dca\logs\train2-dca\weights\best.pt"
# results dir
sample_dir = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\dataset_preprocess\yolo_dca\mask_sample"
missed_dir = os.path.join(sample_dir, "missed_image")
missed_images_dir = os.path.join(missed_dir, "images")
missed_labels_dir = os.path.join(missed_dir, "labels")

class_names = ["MEL", "MNV", "NV"]  # You can modify this list as needed