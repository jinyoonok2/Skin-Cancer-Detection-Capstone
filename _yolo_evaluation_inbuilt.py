from ultralytics import YOLO
import os
from dataset_preprocess.yolo_txt.dataset_path import DATASET_DIR

# train8-epoch100-nonUNK
# train9-epoch80-nonUNK

if __name__ == '__main__':

    # Dataset paths
    dataset_names = ['train9-eval.yaml']
    dataset_paths = [os.path.join(DATASET_DIR, dataset_name) for dataset_name in dataset_names]

    # Model paths
    model_paths = [
        r'C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\detect\train9-epoch80-nonUNK\weights\best.pt',
        # r'.\runs\detect\train10-epoch80-nonUNK\weights\best.pt'
    ]

    # Ensure model_paths and dataset_paths are of the same length
    if len(model_paths) != len(dataset_paths):
        print("The number of models does not match the number of datasets.")
    else:
        for model_path, dataset_path in zip(model_paths, dataset_paths):
            model = YOLO(model_path)
            model.val(data=dataset_path, save_json=True, device=0, plots=True, split='test')
