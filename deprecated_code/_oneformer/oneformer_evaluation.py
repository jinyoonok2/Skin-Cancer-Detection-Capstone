from transformers import AutoModelForUniversalSegmentation
import torch
import matplotlib.pyplot as plt
from one_former_custom_dataload import processor, id2label  # Custom dataloader and processor
import os
import csv
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from PIL import Image

from dataset_preprocess.one_former.dataset_path import TEST_PATH_IMAGES, TEST_PATH_MASKS

id2label_eval = {**id2label, 0: "background"}


def process_images_and_save_results(model, processor, TEST_PATH_IMAGES, TEST_PATH_MASKS, csv_file_path, device):
    """
    Process images, compare predicted and actual classes, and save results to a CSV file.
    If the CSV file already exists, it skips creating a new file and proceeds with the image processing.
    """
    # Ensure the directory for the CSV file exists
    csv_dir = os.path.dirname(csv_file_path)  # Extract directory path from the csv_file_path
    os.makedirs(csv_dir, exist_ok=True)  # Create the directory if it does not exist

    # Ensure the CSV file exists and has the correct header
    file_exists = os.path.exists(csv_file_path)
    if not file_exists:
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Predicted Class', 'Actual Class'])
    else:
        print(f"{csv_file_path} already exists. Skipping file creation and moving to image processing.")

    # Proceed with image processing only if the file was just created or already existed
    if not file_exists or (file_exists and input("File exists. Process images anyway? (y/n): ").lower() == 'y'):
        with open(csv_file_path, 'a', newline='') as csvfile:  # Open in append mode
            csvwriter = csv.writer(csvfile)

            image_files = sorted(os.listdir(TEST_PATH_IMAGES))
            for image_file in image_files:
                image_path = os.path.join(TEST_PATH_IMAGES, image_file)  # Full image path
                image = Image.open(image_path).convert("RGB")  # Load and convert the image
                inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")  # Prepare image

                with torch.no_grad():  # Forward pass
                    outputs = model(**inputs)

                # Postprocessing
                semantic_segmentation = \
                processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                labels_ids = torch.unique(semantic_segmentation).tolist()
                if 0 in labels_ids: labels_ids.remove(0)  # Remove background label ID

                # Count pixels for each label and find the max
                label_pixel_count = {label: (semantic_segmentation == label).sum().item() for label in labels_ids}
                max_label = max(label_pixel_count, key=label_pixel_count.get, default=0)

                # Retrieve the actual class from the mask file
                mask_file_name = image_file.replace('.jpg', '_mask.png')
                mask_path = os.path.join(TEST_PATH_MASKS, mask_file_name)
                mask = Image.open(mask_path).convert('L')
                mask_array = np.array(mask)
                actual_class_id = np.max(mask_array)

                # Write and flush the results
                csvwriter.writerow([max_label, actual_class_id])
                csvfile.flush()

                print(f"Processed {image_file}: Predicted Class = {max_label}, Actual Class = {actual_class_id}")
    else:
        print("Skipping image processing.")

def create_normalized_confusion_matrix(csv_file_path):
    """
    Reads the CSV file to create and display a normalized confusion matrix using Matplotlib.
    """
    predicted_classes = []
    actual_classes = []

    # Read the CSV file
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            predicted_classes.append(int(row[0]))
            actual_classes.append(int(row[1]))

    # Calculate the confusion matrix
    cm = confusion_matrix(actual_classes, predicted_classes, normalize='true')

    # Plotting using Matplotlib and Seaborn for better aesthetics
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=plt.cm.Blues, cbar=False)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the plot
    plt.savefig('oneformer_results/normalized_confusion_matrix.png')
    plt.show()


def print_classification_metrics(csv_file_path):
    predicted_classes = []
    actual_classes = []

    # Read the CSV file
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            predicted_classes.append(int(row[0]))
            actual_classes.append(int(row[1]))

    # Calculate precision, recall, F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(actual_classes, predicted_classes, average=None)

    # Calculate accuracy
    accuracy = accuracy_score(actual_classes, predicted_classes)

    # Calculate averages for precision, recall, and F1 score
    average_precision = np.mean(precision)
    average_recall = np.mean(recall)
    average_f1 = np.mean(f1)

    # Print metrics for each class
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i + 1}: Precision: {p:.2f}, Recall: {r:.2f}, F1 Score: {f:.2f}")

    # Print total averages and accuracy
    print(f"\nTotal Average Precision: {average_precision:.2f}")
    print(f"Total Average Recall: {average_recall:.2f}")
    print(f"Total Average F1 Score: {average_f1:.2f}")
    print(f"Overall Accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)
    model_path = r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\one_former_segment\checkpoint_epoch_16.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.model.is_training = False
    model.eval()

    csv_file_path = 'oneformer_results/oneformer_prediction_comparison.csv'  # Full path to the CSV file

    # Process images and save results to CSV if not already done
    process_images_and_save_results(model, processor, TEST_PATH_IMAGES, TEST_PATH_MASKS, csv_file_path, device)

    # Create and display the normalized confusion matrix
    create_normalized_confusion_matrix(csv_file_path)

    print_classification_metrics(csv_file_path)
