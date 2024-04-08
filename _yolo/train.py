from dataset_preprocess.yolo_detect.dataset_path import YAML_PATH
from ultralytics import YOLO
def main():
    # just normal run
    model = YOLO('yolov8s.pt')
    model.train(data=YAML_PATH, imgsz=512, epochs=100, device=0, name='train1-after-removal', project='./logs', optimizer='AdamW')

if __name__ == '__main__':
    main()\

