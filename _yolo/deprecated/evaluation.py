from ultralytics import YOLO
import os
from pathlib import Path
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import numpy as np
import json


class ModelEvaluation:
    def __init__(self, setting_number):
        self.setting = self.get_settings(setting_number)
        if self.setting == "Settings not found.":
            raise ValueError("Invalid setting number provided.")
        self.load_settings()

    def get_settings(self, setting_number):
        settings = {
            1: {
                'YAML_PATH': r"C:\Jinyoon Projects\datasets\combined_dataset\train9-eval.yaml",
                'MODEL_PATH': r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\detect\train9-epoch80-nonUNK\weights\best.pt",
                'DATASET_DIR': r'C:\Jinyoon Projects\datasets\combined_dataset\train9-test',
                'SAVE_EVAL_PATH': r'C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\_yolo',
            },
            2: {
                'YAML_PATH': r"C:\Jinyoon Projects\datasets\combined_dataset_cascade_mela\data.yaml",
                'MODEL_PATH': r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\detect\train1-mela-epoch100\weights\best.pt",
                'DATASET_DIR': r'C:\Jinyoon Projects\datasets\combined_dataset_cascade_mela\test',
                'SAVE_EVAL_PATH': r'C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\_yolo_cascade_mela',
            },
            3: {
                'YAML_PATH': r"C:\Jinyoon Projects\datasets\combined_dataset_cascade_non_mela\data.yaml",
                'MODEL_PATH': r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\detect\train1-nonmela-epoch100\weights\best.pt",
                'DATASET_DIR': r'C:\Jinyoon Projects\datasets\combined_dataset_cascade_non_mela\test',
                'SAVE_EVAL_PATH': r'C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\_yolo_cascade_non_mela',
            },
            # Add more settings as needed
        }
        return settings.get(setting_number, "Settings not found.")

    def load_settings(self):
        self.YAML_PATH = self.setting['YAML_PATH']
        self.MODEL_PATH = self.setting['MODEL_PATH']
        self.DATASET_DIR = self.setting['DATASET_DIR']
        self.SAVE_EVAL_PATH = self.setting['SAVE_EVAL_PATH']
        self.NO_DETECTION_LABEL = -1
        self.class_names = self.load_class_names(self.YAML_PATH)
        self.model = self.setup_model(self.MODEL_PATH)

    def load_class_names(self, yaml_path):
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data['names']

    # Setup YOLO model
    def setup_model(self, model_path):
        return YOLO(model_path)

    # Compute metrics
    def compute_metrics(self, true_labels, predicted_labels, all_labels):
        precision_per_class = precision_score(true_labels, predicted_labels, labels=all_labels, average=None,
                                              zero_division=0)
        recall_per_class = recall_score(true_labels, predicted_labels, labels=all_labels, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predicted_labels, labels=all_labels, average=None, zero_division=0)
        precision_avg = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        recall_avg = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        f1_avg = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
        return precision_per_class, recall_per_class, f1_per_class, precision_avg, recall_avg, f1_avg

    # Compute metrics from CSV
    def compute_metrics_from_csv(self, csv_path, class_names):
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            predicted_labels = []
            true_labels = []
            for row in reader:
                predicted_labels.append(int(float(row['Prediction'])))
                true_labels.append(int(float(row['Actual'])))

        # Adjusting all_labels to exclude no_detection_label for metrics calculation
        all_labels = list(range(len(class_names)))  # Now excluding no_detection_label

        precision_per_class, recall_per_class, f1_per_class, precision_avg, recall_avg, f1_avg = self.compute_metrics(
            true_labels, predicted_labels, all_labels)

        # Calculate overall accuracy including the 'no_detection' cases
        accuracy_overall = accuracy_score(true_labels, predicted_labels)

        # Calculate per-class accuracy excluding no_detection_label
        class_accuracies = {}
        for label in all_labels:  # Already excludes no_detection_label
            class_specific_true = [1 if label == true_label else 0 for true_label in true_labels]
            class_specific_pred = [1 if label == pred_label else 0 for pred_label in predicted_labels]
            class_accuracies[class_names[label]] = accuracy_score(class_specific_true, class_specific_pred)

        metrics = {
            "per_class": {
                class_names[idx]: {
                    "Precision": precision_per_class[idx],
                    "Recall": recall_per_class[idx],
                    "F1 Score": f1_per_class[idx],
                    "Accuracy": class_accuracies[class_names[idx]]  # No need to check for N/A anymore
                } for idx in range(len(class_names))
            },
            "average": {
                "Precision": precision_avg,
                "Recall": recall_avg,
                "F1 Score": f1_avg,
                "Accuracy": accuracy_overall
            }
        }
        return metrics

    # Save predictions and actual labels to CSV
    def save_predictions_to_csv(self, test_images, test_labels, model, conf_score, csv_path, no_detection_label=-1):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Prediction', 'Actual'])

            results = model.predict(source=test_images, device=0, conf=conf_score, stream=True)
            for result in results:
                basename = Path(result.path).stem
                label_file = os.path.join(test_labels, basename + '.txt')
                with open(label_file, 'r') as file:
                    actual_class_id = int(file.readline().split()[0])
                predicted_class_id = result.boxes.cls[0].item() if result.boxes else no_detection_label
                writer.writerow([predicted_class_id, actual_class_id])

    def save_metrics_to_json(self, all_metrics, save_eval_path, json_file_name):
        json_path = os.path.join(save_eval_path, json_file_name)
        with open(json_path, 'w') as json_file:
            json.dump(all_metrics, json_file, indent=4)

        print(f"All metrics saved to {json_path}")

    # Generate and save confusion matrix
    def generate_and_save_confusion_matrix_from_csv(self, csv_path, all_labels, class_names, matrix_path):
        print(all_labels)
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            predicted_labels = []
            true_labels = []
            for row in reader:
                predicted_labels.append(int(float(row['Prediction'])))
                true_labels.append(int(float(row['Actual'])))

        cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

        # Fix for the normalization process
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = np.divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis],
                                      where=cm.sum(axis=1)[:, np.newaxis] != 0)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                    xticklabels=[class_names.get(l, "No Detection") for l in all_labels],
                    yticklabels=[class_names.get(l, "No Detection") for l in all_labels])

        for _, spine in ax.spines.items():
            spine.set_visible(True)

        plt.title('Normalized Confusion Matrix with Clear Boundaries')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(matrix_path)
        plt.close()

    def evaluate(self, conf_scores):
        YAML_PATH = self.setting['YAML_PATH']
        MODEL_PATH = self.setting['MODEL_PATH']
        DATASET_DIR = self.setting['DATASET_DIR']
        SAVE_EVAL_PATH = self.setting['SAVE_EVAL_PATH']
        NO_DETECTION_LABEL = -1

        class_names = self.load_class_names(YAML_PATH)
        id2label = {i: name for i, name in enumerate(class_names)}
        all_labels = list(range(len(class_names))) + [NO_DETECTION_LABEL]
        model = self.setup_model(MODEL_PATH)

        test_images = os.path.join(DATASET_DIR, 'images')
        test_labels = os.path.join(DATASET_DIR, 'labels')

        all_metrics = {}
        for conf in conf_scores:
            csv_path = os.path.join(SAVE_EVAL_PATH, f'predictions_and_actuals_conf_{conf}.csv')
            if not os.path.exists(csv_path):
                self.save_predictions_to_csv(test_images, test_labels, model, conf, csv_path, NO_DETECTION_LABEL)
                print(f"Predictions and actual labels saved to {csv_path}")

            # Compute metrics and generate confusion matrix regardless of CSV creation
            metrics = self.compute_metrics_from_csv(csv_path, class_names)
            all_metrics[f"conf_{conf}"] = metrics
            matrix_path = os.path.join(SAVE_EVAL_PATH, f'confusion_matrix_conf_{conf}.png')
            self.generate_and_save_confusion_matrix_from_csv(csv_path, all_labels, id2label, matrix_path)
            print(f"Confusion matrix saved to {matrix_path}")

        # Save evaluation metrics after collecting for all confidences
        json_file_name = "evaluation_metrics.json"
        self.save_metrics_to_json(all_metrics, SAVE_EVAL_PATH, json_file_name)

    def yolo_builtin_evaluation(self):
        print(f"Evaluating model: {self.MODEL_PATH} on dataset: {self.YAML_PATH}")
        self.model.val(data=self.YAML_PATH, save_json=True, device=0, plots=True, split='test')

def main(setting_number=1):
    conf_scores = [0.25, 0.35, 0.459, 0.50, 0.65]

    model_evaluator = ModelEvaluation(setting_number)
    model_evaluator.evaluate(conf_scores)
    model_evaluator.yolo_builtin_evaluation()

if __name__ == "__main__":
    main(1)






