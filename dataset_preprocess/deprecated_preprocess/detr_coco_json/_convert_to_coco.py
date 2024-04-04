import json
import glob
import yaml
from PIL import Image
import shutil
from coco_dataset_path import *

def yolo_to_coco(yolo_directory, output_file, data_yaml):
    # Read class names from data.yaml
    with open(data_yaml, 'r') as file:
        categories = yaml.safe_load(file)['names']
    category_dict = {name: i for i, name in enumerate(categories)}

    # Initialize COCO dataset structure
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for name, i in category_dict.items()]
    }

    # Image ID counter
    image_id_counter = 1
    annotation_id = 1

    # Process each image and annotation file
    for image_path in glob.glob(os.path.join(yolo_directory, "images", "*.jpg")):
        image = Image.open(image_path)
        width, height = image.size

        # Add image info to COCO dataset with numeric ID
        coco_dataset['images'].append({
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "id": image_id_counter
        })

        # Process corresponding annotation file
        annotation_file = os.path.join(yolo_directory, "labels", os.path.basename(image_path).split('.')[0] + ".txt")
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    category_id, segmentation_coords = int(parts[0]), parts[1:]
                    segmentation_coords = [float(coord) for coord in segmentation_coords]

                    # Convert segmentation coordinates from relative to absolute
                    segmentation_coords = [segmentation_coords[i] * width if i % 2 == 0 else segmentation_coords[i] * height for i in range(len(segmentation_coords))]

                    # Convert to bbox format for COCO
                    x_coords = segmentation_coords[0::2]
                    y_coords = segmentation_coords[1::2]
                    x_min, y_min, x_max, y_max = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                    # Add annotation info to COCO dataset
                    coco_dataset['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id_counter,
                        "category_id": category_id,
                        "bbox": bbox,
                        "segmentation": [segmentation_coords],
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0
                    })
                    annotation_id += 1

        # Increment the image ID counter
        image_id_counter += 1

    # Write COCO dataset to JSON file
    with open(output_file, 'w') as file:
        json.dump(coco_dataset, file, indent=4)


def move_images_to_main_dir(base_directory):
    images_directory = os.path.join(base_directory, "images")

    if not os.path.exists(images_directory):
        print(f"No 'images' directory found in {base_directory}.")
        return

    for image_file in os.listdir(images_directory):
        source_path = os.path.join(images_directory, image_file)
        destination_path = os.path.join(base_directory, image_file)

        # Move each image to the main directory
        shutil.move(source_path, destination_path)

    # Optionally, remove the now-empty 'images' directory
    os.rmdir(images_directory)
    print(f"Images moved from '{images_directory}' to '{base_directory}'.")


# Example usage with directory-specific output file paths
yolo_to_coco(yolo_directory=COCO_TRAIN_PATH, output_file=os.path.join(COCO_TRAIN_PATH, 'annotations.coco.json'), data_yaml=COCO_YAML_PATH)
yolo_to_coco(yolo_directory=COCO_VALID_PATH, output_file=os.path.join(COCO_VALID_PATH, 'annotations.coco.json'), data_yaml=COCO_YAML_PATH)
yolo_to_coco(yolo_directory=COCO_TEST_PATH, output_file=os.path.join(COCO_TEST_PATH, 'annotations.coco.json'), data_yaml=COCO_YAML_PATH)

# Example usage
move_images_to_main_dir(COCO_TRAIN_PATH)
move_images_to_main_dir(COCO_VALID_PATH)
move_images_to_main_dir(COCO_TEST_PATH)