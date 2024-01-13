from ultralytics import YOLO
import torch



def main():
    print(torch.cuda.is_available())
    # Train the model
    model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights
    model.train(data='datasets/HAM10000/data.yaml', epochs=1, imgsz=256, device=0)
    # model = YOLO('runs/detect/train2/weights/best.pt')
    # results = model.predict(data='datasets/mel_test1.jpg')


if __name__ == '__main__':
    main()
