from dataset_preprocess.yolo_hair.dataset_path import DATASET_DIR
from ultralytics import YOLO
def main():
    # just normal run
    model = YOLO('yolov8s-cls.pt')
    model.train(data=DATASET_DIR, imgsz=512, epochs=50, device=0, name='train1-hair', project='./logs')

if __name__ == '__main__':
    main()