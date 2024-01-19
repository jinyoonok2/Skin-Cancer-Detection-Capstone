from ultralytics import YOLO

import os
import torch
import gc
import glob
import shutil
import csv
import yaml
import pandas as pd
import yaml

class ModelHandler:
    def __init__(self, data_path, model_path=None):
        self.data_path = data_path
        self.model_path = model_path
        self.cls_map = self.extract_cls_map(self.data_path)

    def train(self, epoch=100, imgsz=256):
        # train with current model_path, then change the model path to EXP_NAME model.
        model = YOLO(self.model_path)
        model.train(data=self.data_path, name=self.data_path, epochs=epoch, imgsz=imgsz, device=0)

        # Update model_path to point to the newly trained model
        model_path = f"runs/segment/{self.data_path}/weights/best.pt"
        self.model_path = os.path.join(os.getcwd(), model_path)
        # unfinished. please correct it to train on right format of datapath later on

    def extract_cls_map(self, yaml_path):
        # Initialize an empty dictionary to store the mapping
        cls_map = {}

        # Open the YAML file at the given path
        with open(yaml_path, 'r') as file:
            # Load the contents of the file
            content = yaml.safe_load(file)

            # Check if 'names' key exists in the YAML content and it's a list
            if 'names' in content and isinstance(content['names'], list):
                # Iterate over the diseases listed under the 'names' key, assigning each one an integer index
                for index, disease in enumerate(content['names']):
                    cls_map[index] = disease

        return cls_map

    def classify_data(self, img_path, table_path, dataset_version):
        # Create class directories according to the cls_map
        for cls_idx, cls_name in self.cls_map.items():
            os.makedirs(os.path.join(img_path, cls_name), exist_ok=True)

        if dataset_version == 'ISIC2019':
            # Read the CSV file
            data = pd.read_csv(table_path)
            # Get the list of columns (diseases) in the CSV file
            csv_columns = data.columns.tolist()

            # Check if 'NV' column exists and change it to 'MNV' in cls_map
            if 'NV' in csv_columns and 'MNV' in self.cls_map.values():
                csv_columns[csv_columns.index('NV')] = 'MNV'
                # Also update the data frame to reflect this change
                data.rename(columns={'NV': 'MNV'}, inplace=True)

            for index, row in data.iterrows():
                # Get the image name
                image_name = row['image'] + '.jpg'
                # Determine the disease type for the image and move it
                for disease in self.cls_map.values():
                    if disease in csv_columns and row[disease] == 1:
                        # Define the new image name with the class appended
                        new_image_name = f"{row['image']}_{disease}.jpg"
                        # Move the image to the corresponding directory with the new name
                        source = os.path.join(img_path, image_name)
                        destination = os.path.join(img_path, disease, new_image_name)
                        os.rename(source, destination)
                        break

        elif dataset_version == 'ISIC2020':
            # Read the CSV file
            data = pd.read_csv(table_path)
            # Get the list of unique diagnoses in the CSV file
            unique_diagnoses = data['diagnosis'].unique().tolist()
            for index, row in data.iterrows():
                # Get the image name and diagnosis
                image_name = row['image_name'] + '.jpg'
                diagnosis = row['diagnosis']
                # Check if the diagnosis is one of the classes we have
                if diagnosis in self.cls_map.values() and diagnosis in unique_diagnoses:
                    # Define the new image name with the class appended
                    new_image_name = f"{row['image_name']}_{diagnosis}.jpg"
                    # Move the image to the corresponding directory with the new name
                    source = os.path.join(img_path, image_name)
                    destination = os.path.join(img_path, diagnosis, new_image_name)
                    os.rename(source, destination)

        return

    def infer(self, img_path, batch_size=50):
        classes = list(self.cls_map.values())
        model = YOLO(self.model_path)

        for cls in classes:
            cls_path = os.path.join(img_path, cls)
            if not os.path.exists(cls_path):
                print(f"Directory for class '{cls}' not found, skipping...")
                continue

            img_files = glob.glob(os.path.join(cls_path, "*.jpg"))

            # Process in batches
            for i in range(0, len(img_files), batch_size):
                batch_files = img_files[i:i + batch_size]
                results = model.predict(batch_files, conf=0.5, device=0)

                # Skip this batch if no results are returned
                if results is None:
                    continue

                label_dest = os.path.join(img_path, cls, 'labels')
                img_dest = os.path.join(img_path, cls, 'images')
                os.makedirs(label_dest, exist_ok=True)
                os.makedirs(img_dest, exist_ok=True)

                for img_file, result in zip(batch_files, results):
                    # Proceeding under the assumption that result is not None
                    base_name = os.path.basename(img_file)
                    label_file_name = os.path.splitext(base_name)[0] + '.txt'
                    txt_file_path = os.path.join(label_dest, label_file_name)

                    result.save_txt(txt_file=txt_file_path)
                    shutil.move(img_file, os.path.join(img_dest, base_name))

        self.correct_results(img_path)

    def correct_results(self, img_path):
        for cls_idx, cls_name in self.cls_map.items():
            label_dir = os.path.join(img_path, cls_name, 'labels')

            # Check if the labels directory exists
            if not os.path.isdir(label_dir):
                print(f"No label directory found for class {cls_name}, skipping correction.")
                continue

            for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
                with open(label_file, 'r') as file:
                    lines = file.readlines()

                with open(label_file, 'w') as file:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) > 1:  # Ensure the line has enough parts to be a valid label
                            # Correct the class index
                            parts[0] = str(cls_idx)
                            # Write the corrected line
                            file.write(" ".join(parts) + "\n")

    def val(self, TEST_DATA_PATH):
        model = YOLO(self.model_path)
        metrics = model.val(data=TEST_DATA_PATH, device=0)
        torch.cuda.empty_cache()
        gc.collect()
        return metrics
