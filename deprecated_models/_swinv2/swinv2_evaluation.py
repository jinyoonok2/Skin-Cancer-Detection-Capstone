from transformers import AutoModelForImageClassification
import torch
from _swinv2_custom_dataload import image_processor
import os
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import csv

checkpoint_directory = r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\swinv2-tiny-patch4-window8-256-finetuned-cd\checkpoint-10400"

# Load the trained model
model = AutoModelForImageClassification.from_pretrained(checkpoint_directory)

valid_images_mlc = r"C:\Jinyoon Projects\datasets\combined_dataset_swin_split\valid\MLC"
valid_images_non_mlc = r"C:\Jinyoon Projects\datasets\combined_dataset_swin_split\valid\NON-MLC"


def process_predictions_and_flush_csv(model, image_processor, csv_file_path, valid_images_mlc, valid_images_non_mlc):
    # Check if CSV already exists
    if os.path.exists(csv_file_path):
        print("CSV file already exists. Skipping the prediction process.")
        return

    # Initialize CSV file with headers
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["predicted", "actual"])

    predictions = []
    actuals = []

    # Function to process a single directory and append each prediction to the CSV
    def process_directory(directory, actual_label):
        for image_name in os.listdir(directory):
            image_path = os.path.join(directory, image_name)
            image = Image.open(image_path)

            # Preprocess the image
            encoding = image_processor(image.convert("RGB"), return_tensors="pt")

            # Perform inference
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits

            # Get the predicted class index and label
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]

            # Append prediction to CSV
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([predicted_label, actual_label])

            predictions.append(predicted_label)
            actuals.append(actual_label)

    # Process each directory
    process_directory(valid_images_mlc, 'MLC')
    process_directory(valid_images_non_mlc, 'NON-MLC')

    print(f"Predictions flushed to {csv_file_path}")

    # After all predictions are done, generate and plot confusion matrix
    labels = ['MLC', 'NON-MLC']
    cm = confusion_matrix(actuals, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
    disp_normalized.plot(cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.show()

def generate_confusion_matrix_from_csv(csv_file_path):
    # Load predictions from CSV
    data = pd.read_csv(csv_file_path)

    # Extract predictions and actual labels
    predictions = data['predicted']
    actuals = data['actual']

    # Generate confusion matrix
    labels = ['MLC', 'NON-MLC']
    cm = confusion_matrix(actuals, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Generate and plot normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
    plt.figure(figsize=(10, 7))
    disp_normalized.plot(cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.show()

def generate_and_save_confusion_matrix(csv_file_path):
    # Load predictions from CSV
    data = pd.read_csv(csv_file_path)

    # Extract predictions and actual labels
    predictions = data['predicted']
    actuals = data['actual']

    # Generate confusion matrix
    labels = ['MLC', 'NON-MLC']
    cm = confusion_matrix(actuals, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Define the base path for saving images
    base_path = os.path.dirname(csv_file_path)

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 7))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(base_path, 'confusion_matrix.png'))  # Save the figure
    plt.show()  # Show the plot

    # Generate and plot normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
    plt.figure(figsize=(10, 7))
    disp_normalized.plot(cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(base_path, 'normalized_confusion_matrix.png'))  # Save the figure
    plt.show()  # Show the plot


def main():
    csv_file_path = r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\swinv2-tiny-patch4-window8-256-finetuned-cd\swinv2_predictions.csv"  # Update this path accordingly
    process_predictions_and_flush_csv(model, image_processor, csv_file_path, valid_images_mlc, valid_images_non_mlc)
    generate_and_save_confusion_matrix(csv_file_path)

if __name__ == '__main__':
    main()