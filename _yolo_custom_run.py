from _yolo_custom_trainer import *
from ultralytics import YOLO
from dataset_preprocess.yolo_txt._train_magnification import magnification_factors
from dataset_preprocess.yolo_txt.dataset_path import *

class_augment_check = {i: magnification_factors[class_name] > 1 for i, class_name in enumerate(magnification_factors)}


# magnification_factors = {
#     'MEL': 4, 'MNV': 2, 'NV': 2, 'BCC': 4, 'AK': 16,
#     'BKL': 8, 'DF': 16, 'VASC': 8, 'SCC': 16, 'UNK': 1
# }


def main():

    # original trainer on HAM10000
    # args = dict(model='yolov8s.pt', data='datasets/HAM10000/data.yaml', imgsz=256, epochs=1)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    # model.train(data='datasets/HAM10000/data.yaml', epochs=3, imgsz=256)

    # 1. custom trainer on combined
    args = dict(model='yolov8s.pt', data=YAML_PATH, imgsz=256, epochs=100, device=0)

    # 2. resume run*
    # args = dict(model='runs/detect/train2/weights/last.pt', data=YAML_PATH, imgsz=256, epochs=100, device=0, resume=True)
    trainer = CustomTrainer(overrides=args)
    trainer.train()

    # model test
    # model = YOLO("runs/detect/train2/weights/best.pt")
    # model.val(data='datasets\combined_dataset\data.yaml', plots=True)



if __name__ == "__main__":
    main()