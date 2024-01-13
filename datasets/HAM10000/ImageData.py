import csv
from PIL import Image
import os
import cv2
import yaml
import shutil

class ImageData:
    def __init__(self, image_dir, mask_dir, label_dir, csv_path):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.csv_path = csv_path
        self.disease_types = self.extract_disease_types()

    def extract_disease_types(self):
        with open(self.csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            # Convert each disease type to uppercase before adding to the set
            disease_types = set(row['dx'].upper() for row in reader)
        print(f"Unique disease types: {disease_types}")
        return disease_types

    def create_data_yaml(self):
        # Ensure the disease types are capitalized and sorted
        disease_types = sorted([disease.upper() for disease in self.disease_types])

        # Prepare the dictionary to be written into the YAML file
        yaml_content = {
            'train': './train/images',
            'val': './valid/images',
            'test': './test/images',  # Assuming you have a test set, adjust if not
            'nc': len(disease_types),
            'names': disease_types
        }

        # Define the path for the YAML file
        yaml_path = 'data.yaml'

        # Write the YAML content to the file
        with open(yaml_path, 'w') as file:
            yaml.dump(yaml_content, file, sort_keys=False)
        print(f"Created data.yaml at {yaml_path}")

    def resize_images(self):
        for directory, dir_type in [(self.image_dir, 'image'), (self.mask_dir, 'mask')]:
            print(f"Resizing {dir_type} files in {directory}...")
            if not os.path.exists(directory):
                print(f"Directory {directory} does not exist.")
                continue

            for filename in os.listdir(directory):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        file_path = os.path.join(directory, filename)
                        with Image.open(file_path) as img:
                            img = img.resize((256, 256), Image.LANCZOS)
                            img.save(file_path)
                            print(f"Resized {filename}")
                    except Exception as e:
                        print(f"Error resizing {filename}: {e}")

    def remove_seg_text(self):
        directory = self.mask_dir  # Using class variable
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            return

        for filename in os.listdir(directory):
            if '_segmentation' in filename:
                try:
                    file_path = os.path.join(directory, filename)
                    new_filename = filename.replace('_segmentation', '')
                    new_file_path = os.path.join(directory, new_filename)
                    os.rename(file_path, new_file_path)
                    print(f"Renamed {filename} to {new_filename}")
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")

    def rename_images(self):
        current_dir = os.getcwd()
        for class_name in self.disease_types:
            class_dir = os.path.join(current_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.txt')):
                        base_name = os.path.splitext(filename)[0]
                        ext = os.path.splitext(filename)[1]  # Get the file extension
                        new_filename = f"{base_name}_{class_name}{ext}"
                        os.rename(os.path.join(class_dir, filename),
                                  os.path.join(class_dir, new_filename))
                print(f"Renamed files in {class_dir}")

    def extract_polygons(self, image_path, epsilon_factor=0.001):
        image = cv2.imread(image_path, 0)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []

        for contour in contours:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            norm_poly = [(point[0][0] / image.shape[1], point[0][1] / image.shape[0]) for point in approx]
            polygons.append(norm_poly)

        return polygons

    def map_disease_types_to_indices(self):
        # Ensure disease types are sorted consistently
        sorted_diseases = sorted(list(self.disease_types))
        return {disease: index for index, disease in enumerate(sorted_diseases)}

    def save_annotations(self, polygons, save_path, class_id):
        with open(save_path, 'w') as file:
            for poly in polygons:
                line = f"{class_id} " + ' '.join([f"{x:.6f} {y:.6f}" for x, y in poly]) + '\n'
                file.write(line)

    def process_images_for_annotations(self, epsilon_factor=0.005):
        disease_to_index = self.map_disease_types_to_indices()

        # Create a mapping from image_id to disease type from the CSV
        image_id_to_dx = {}
        with open(self.csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_id_to_dx[row['image_id']] = row['dx'].upper()

        # Process each file in the mask directory
        for filename in os.listdir(self.mask_dir):
            if filename.lower().endswith('.png'):
                base_name = os.path.splitext(filename)[0]
                class_id = disease_to_index.get(image_id_to_dx.get(base_name, '').upper(), 0)

                image_path = os.path.join(self.mask_dir, filename)
                save_path = os.path.join(self.label_dir, f"{base_name}.txt")

                polygons = self.extract_polygons(image_path, epsilon_factor)
                self.save_annotations(polygons, save_path, class_id)
                print(f"Processed {filename} with class_id {class_id}")

    def filter_annotations(self):
        # Iterate through all files in the directory
        for filename in os.listdir(self.label_dir):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(self.label_dir, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # Filter out annotations with four or fewer coordinates
                filtered_lines = [line for line in lines if len(line.strip().split(' ')) > 5]

                # Write the filtered annotations back to the file
                with open(file_path, 'w') as file:
                    file.writelines(filtered_lines)
                print(f"Processed {filename}")
    def split_to_class(self):
        # Read CSV and map image IDs to their respective disease types
        image_to_dx = {}
        with open(self.csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_to_dx[row['image_id']] = row['dx'].upper()

        # Create directories for each disease type in the current directory
        for disease in self.disease_types:
            os.makedirs(os.path.join(os.getcwd(), disease), exist_ok=True)

        # Move corresponding images and labels to their class directory
        for img_id, dx in image_to_dx.items():
            src_img_path = os.path.join(self.image_dir, img_id + '.jpg')
            src_label_path = os.path.join(self.label_dir, img_id + '.txt')
            dest_dir = os.path.join(os.getcwd(), dx)

            if os.path.exists(src_img_path):
                dest_img_path = os.path.join(dest_dir, img_id + '.jpg')
                shutil.move(src_img_path, dest_img_path)

            if os.path.exists(src_label_path):
                dest_label_path = os.path.join(dest_dir, img_id + '.txt')
                shutil.move(src_label_path, dest_label_path)