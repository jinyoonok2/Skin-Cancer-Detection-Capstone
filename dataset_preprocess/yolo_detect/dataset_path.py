import os

# YOLO dataset
DATASET_DIR = r'C:\Jinyoon_Projects\datasets\combined_dataset'
DEFAULT_BEFORE_PATH = os.path.join(DATASET_DIR, 'default_before')
DEFAULT_AFTER_PATH = os.path.join(DATASET_DIR, 'default_after')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
TRAIN_PATH = os.path.join(DATASET_DIR, 'train')
VALID_PATH = os.path.join(DATASET_DIR, 'valid')
TEST_PATH = os.path.join(DATASET_DIR, 'test')


relabel_dir = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\dataset_preprocess\yolo_detect\removal_results"
# Path for relabel
hair_images_save_dir = os.path.join(relabel_dir, "hair_relabel", "images")
hair_original_save_dir = os.path.join(relabel_dir, "hair_relabel", "original")
hair_labels_save_dir = os.path.join(relabel_dir, "hair_relabel", "labels")

hair_dca_images_save_dir = os.path.join(relabel_dir, "hair_dca_relabel", "images")
hair_dca_original_save_dir = os.path.join(relabel_dir, "hair_dca_relabel", "original")
hair_dca_labels_save_dir = os.path.join(relabel_dir, "hair_dca_relabel", "labels")

dca_save_dir = os.path.join(relabel_dir, "dca", "images")
dca_original_save_dir = os.path.join(relabel_dir, "dca", "original")
# Path for plot
hair_plot_save_dir = os.path.join(relabel_dir, "hair_before_after")
hair_dca_plot_save_dir = os.path.join(relabel_dir, "hair_dca_before_after")
dca_plot_save_dir = os.path.join(relabel_dir, "dca_before_after")

# EigenCAM result dir
eigen_cam_dir = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\dataset_preprocess\yolo_detect\cam_results"

# EigenCAM images directory path
CAM_comparison_before_image_dir = {
    "hair": hair_original_save_dir,
    "hair_dca": hair_dca_original_save_dir,
    "dca": dca_original_save_dir
}
CAM_comparison_after_image_dir = {
    "hair": hair_images_save_dir,
    "hair_dca": hair_dca_images_save_dir,
    "dca": dca_save_dir
}
# EigenCAM output directory path
CAM_comparison_output_dir = {
    "hair": os.path.join(eigen_cam_dir, "hair"),
    "hair_dca": os.path.join(eigen_cam_dir, "hair_dca"),
    "dca": os.path.join(eigen_cam_dir, "dca")
}

# EigenCAM model path
before_model_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo\logs\train1-before-removal-seed42\weights\best.pt"
after_model_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo\logs\train1-after-removal-seed42\weights\best.pt"