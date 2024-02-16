from _yolo_custom_trainer import *
from ultralytics import YOLO
# from dataset_preprocess.yolo_txt._train_magnification import magnification_factors
# from dataset_preprocess.yolo_txt.dataset_path import *
from dataset_preprocess.yolo_txt_cascade.dataset_path import *




# magnification_factors = {
#     'MEL': 4, 'MNV': 2, 'NV': 2, 'BCC': 4, 'AK': 16,
#     'BKL': 8, 'DF': 16, 'VASC': 8, 'SCC': 16, 'UNK': 1
# }


def main():

    # Default Train
    # model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights
    # model.train(data=YAML_PATH, epochs=50, imgsz=256, name='train1-cascade-epoch50-default', workers=4)

    # model = YOLO(r'C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\detect\train1-cascade-epoch50-default\weights\last.pt')
    # model.train(data=YAML_PATH, epochs=50, imgsz=256, workers=4, device=0, resume=True)

    # 1. custom trainer on combined
    args = dict(model='yolov8s.pt', data=YAML_PATH, imgsz=256, epochs=50, device=0, name='train2-cascade-epoch50', patience=10)

    # 2. resume run*
    # args = dict(model='runs/detect/train1-cascade-epoch50/weights/last.pt', data=YAML_PATH, imgsz=256, epochs=50, device=0, patience=10, resume=True)
    trainer = CustomTrainer(overrides=args)
    trainer.train()

    # # model test
    # model = YOLO("runs/detect/train6-epoch150-nonUNK/weights/best.pt")
    # model.val(data=r"C:\Jinyoon Projects\datasets\combined_dataset\data.yaml", plots=True, split='test')



if __name__ == "__main__":
    main()