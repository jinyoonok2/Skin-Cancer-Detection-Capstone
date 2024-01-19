import argparse
from model_handler import ModelHandler
import torch
import gc
from ultralytics import YOLO

def main():
    # model = YOLO('yolov8s-seg.pt')
    # model.train(data='datasets\HAM10000\data.yaml', epochs = 25, imgsz=256, device =0)
    # model.val(data='datasets\HAM10000\data.yaml', imgsz=256, split='test', save_hybrid=True, conf=0.75)
    model = YOLO('runs/segment/train/weights/best.pt')
    results = model.predict(['datasets/HAM10000/active_learning/MEL/ISIC_0024310-MEL.jpg', 'datasets/HAM10000/active_learning/BKL/ISIC_0024324-BKL.jpg'], save_txt=True, conf=0.5, save_dir='test')
    for result in results:
        result.save_txt(txt_file='datasets/HAM10000/ISIC_0024324-BKL.txt')
        print(result.boxes.conf)
        print(result.path)



if __name__ == '__main__':
    main()

# python manual_train.py datasets\HAM10000\data.yaml