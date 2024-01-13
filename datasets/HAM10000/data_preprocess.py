import argparse
from ImageData import ImageData  # Make sure to import your ImageData class
import sys
import os
def main():
    parser = argparse.ArgumentParser(description='Process images for YOLO annotations.')
    parser.add_argument('image_dir', help='Directory path of the image files.')
    parser.add_argument('mask_dir', help='Directory path of the mask files.')
    parser.add_argument('label_dir', help='Directory path where labels should be saved.')
    parser.add_argument('csv_path', help='Path to the CSV file with image information.')

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of the ImageData class with the provided paths
    image_data = ImageData(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        label_dir=args.label_dir,
        csv_path=args.csv_path
    )

    # Ensure label directory exists
    os.makedirs(image_data.label_dir, exist_ok=True)
    print(f"Ensured label directory exists at {image_data.label_dir}")

    # Remove '_segmentation' from mask filenames
    print("Removing segmentation text from mask filenames...")
    image_data.remove_seg_text()

    # Resize images in both the image and mask directories
    print("Resizing images in the image and mask directories...")
    image_data.resize_images()

    # Process the images in the mask directory and convert to YOLO format
    print("Creating YOLO annotations from mask images...")
    image_data.process_images_for_annotations()
    image_data.filter_annotations()

    # Split images and labels into class directories
    print("Splitting images and labels into class directories...")
    image_data.split_to_class()

    # Rename images and labels according to their class directories
    print("Renaming images and labels according to their class directories...")
    image_data.rename_images()

    # Create the YAML file
    print("Creating data.yaml file...")
    image_data.create_data_yaml()

if __name__ == '__main__':
    main()

# python data_preprocess.py data_images data_masks data_labels HAM10000_metadata.csv
