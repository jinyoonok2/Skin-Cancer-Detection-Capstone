from ultralytics import YOLO
import os
from collections import defaultdict

def process_results(results):
    # Class names dictionary
    names = {0: 'MEL', 1: 'MNV', 2: 'NV', 3: 'BCC', 4: 'AK', 5: 'BKL', 6: 'DF', 7: 'VASC', 8: 'SCC', 9: 'UNK'}

    # Inverse mapping of names for quick lookup
    name_to_num = {v: k for k, v in names.items()}

    # Dictionary to hold the count of each actual class predicted as each class
    class_counts = defaultdict(lambda: defaultdict(int))

    for result in results:
        # Extracting the class name from the file path
        _, filename = os.path.split(result.path)
        actual_class_name = filename.rsplit('_', 1)[1].split('.')[0]
        actual_class_num = name_to_num[actual_class_name]

        # Getting the predicted class number(s)
        predicted_classes = result.boxes.cls.cpu().tolist()

        # Tallying the results
        for predicted_class_num in predicted_classes:
            class_counts[actual_class_num][int(predicted_class_num)] += 1

    # Calculating percentages
    percentages = defaultdict(dict)
    for actual_class, counts in class_counts.items():
        total = sum(counts.values())
        for predicted_class, count in counts.items():
            percentages[names[actual_class]][names[predicted_class]] = (count / total) * 100

    return percentages

def main():
    model = YOLO("runs/detect/train2/weights/best.pt")
    # model.predict(source= 'datasets/combined_dataset/test/images', conf=0.5, save=True, save_conf=True, save_txt=True)
    results = model(source='datasets/combined_dataset/valid/images', stream=True)
    # Assuming results is a list of prediction results
    percentages = process_results(results)
    # Print the percentages
    for actual_class, preds in percentages.items():
        print(f"Percentages for {actual_class}:")
        for predicted_class, percentage in preds.items():
            print(f"  {predicted_class}: {percentage:.2f}%")
        print()

if __name__ == "__main__":
    main()