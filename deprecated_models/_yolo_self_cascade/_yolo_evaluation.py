from ultralytics import YOLO
import os
from pathlib import Path
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_preprocess.yolo_detect.dataset_path import DATASET_DIR, YAML_PATH
import yaml
import numpy as np

# Load class names from YAML
with open(YAML_PATH, 'r') as file:
    yaml_data = yaml.safe_load(file)
    class_names = yaml_data['names']  # Adjust if the key for class names is different in your YAML file

# Create a mapping from ID to label
id2label = {i: name for i, name in enumerate(class_names)}

# Constants
NO_DETECTION_LABEL = -1  # Define a unique label for 'no detection' class

# Paths
test_path = os.path.join(DATASET_DIR, 'train9-test')
test_images = os.path.join(test_path, 'images')
test_labels = os.path.join(test_path, 'labels')

model_path = r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\detect\train9-epoch80-nonUNK\weights\best.pt"
model = YOLO(model_path)

# Define your actual class labels (assuming they start from 0 to N-1)
actual_labels = list(range(len(class_names)))  # N is the number of classes

# Include NO_DETECTION_LABEL in your labels for confusion matrix and metrics calculation
all_labels = actual_labels + [NO_DETECTION_LABEL]

conf_scores = [0.25, 0.35, 0.459, 0.50, 0.65]
for conf in conf_scores:
    predicted_labels = []
    true_labels = []

    results = model.predict(source=test_images, device=0, conf=conf, stream=True)
    for result in results:
        basename = Path(result.path).stem
        label_file = os.path.join(test_labels, basename + '.txt')

        with open(label_file, 'r') as file:
            first_class_id = int(file.readline().split()[0])
            true_labels.append(first_class_id)

        if result.boxes:
            predicted_class_id = result.boxes.cls[0].item()
            predicted_labels.append(predicted_class_id)
        else:
            predicted_labels.append(NO_DETECTION_LABEL)  # No detection case

    # Compute metrics per class
    precision_per_class = precision_score(true_labels, predicted_labels, labels=all_labels, average=None,
                                          zero_division=0)
    recall_per_class = recall_score(true_labels, predicted_labels, labels=all_labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predicted_labels, labels=all_labels, average=None, zero_division=0)

    # Compute average metrics
    precision_avg = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_avg = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_avg = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    save_path = r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\detect\train9-epoch80-nonUNK\save_eval"
    os.makedirs(save_path, exist_ok=True)
    # CSV Path for this confidence
    csv_path = os.path.join(save_path, f'evaluation_metrics_conf_{conf}.csv')

    # Write metrics to CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1 Score'])

        for idx in all_labels:
            pr = precision_per_class[idx] if idx != NO_DETECTION_LABEL else "N/A"
            re = recall_per_class[idx] if idx != NO_DETECTION_LABEL else "N/A"
            f1 = f1_per_class[idx] if idx != NO_DETECTION_LABEL else "N/A"
            class_name = id2label.get(idx, "No Detection")  # Get the class name using id2label
            writer.writerow([class_name, pr, re, f1])  # Write class name instead of idx

        # Write average scores in the end
        writer.writerow(['Average', precision_avg, recall_avg, f1_avg])

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

    # Normalize the confusion matrix
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = np.true_divide(cm, cm_sum)
        cm_normalized[cm_sum == 0] = 0  # Set NaNs to 0

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues',
                xticklabels=[id2label.get(l, "No Detection") for l in all_labels],
                yticklabels=[id2label.get(l, "No Detection") for l in all_labels])
    plt.title(f'Normalized Confusion Matrix at Confidence {conf}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the normalized confusion matrix as PNG
    matrix_path = os.path.join(save_path, f'normalized_confusion_matrix_conf_{conf}.png')
    plt.savefig(matrix_path)
    plt.close()

    print(f"Confidence: {conf}, Metrics saved to CSV, Confusion Matrix saved as PNG")

