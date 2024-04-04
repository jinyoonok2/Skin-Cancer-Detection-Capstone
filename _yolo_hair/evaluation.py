from dataset_preprocess.yolo_hair.dataset_path import YAML_PATH
from ultralytics import YOLO
def main():
    model_path = r"C:\Jinyoon_Projects\0_Skin-Cancer-Detection-Capstone\_yolo_hair\logs\train1-hair\weights\best.pt"
    model = YOLO(model_path)
    model.val(data=YAML_PATH, project="./logs")
    model.val(data=YAML_PATH, split='test', project="./logs")

if __name__ == '__main__':
    main()