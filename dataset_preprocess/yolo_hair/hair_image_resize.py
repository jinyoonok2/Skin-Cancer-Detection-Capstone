from PIL import Image
import os
from dataset_preprocess.yolo_hair.dataset_path import HAIR_PATH


def resize_images_inplace(source_dir, size=(512, 512)):
    """
    Resize images in source_dir to the specified size and overwrite them.

    :param source_dir: The directory containing the class folders with images.
    :param size: New size of the images, default is (256, 256).
    """
    # Iterate through each class folder in the source directory
    for class_folder in os.listdir(source_dir):
        class_source_dir = os.path.join(source_dir, class_folder)

        # Iterate through each image in the class folder
        for image_file in os.listdir(class_source_dir):
            if os.path.isfile(os.path.join(class_source_dir, image_file)):
                try:
                    # Open the image
                    with Image.open(os.path.join(class_source_dir, image_file)) as img:
                        # Resize the image
                        img_resized = img.resize(size, Image.Resampling.LANCZOS)

                        # Overwrite the original image with the resized image
                        img_resized.save(os.path.join(class_source_dir, image_file))
                        print(f"Resized and saved {image_file} in place.")
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")


# Example usage
resize_images_inplace(HAIR_PATH)
