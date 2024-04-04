from dataset_preprocess.yolo_dca.dataset_path import YAML_PATH
from ultralytics import YOLO
def main():
    # just normal run
    model = YOLO('yolov8s-seg.pt')
    model.train(data=YAML_PATH, imgsz=512, epochs=100, device=0, name='train1-dca', project='./logs', single_cls=True)

if __name__ == '__main__':
    main()