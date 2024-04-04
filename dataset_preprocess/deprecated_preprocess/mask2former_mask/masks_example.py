from datasets import load_dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import wget

# Ensure the output directory exists
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)


# Load the dataset
def load_data():
    dataset = load_dataset("scene_parse_150", "instance_segmentation")
    return dataset


# Load and process the id-label mapping
def load_id_label_mapping():
    data = pd.read_csv('instanceInfo100_train.txt', sep='\t', header=0, on_bad_lines='skip')
    id2label = {id: label.strip() for id, label in enumerate(data["Object Names"])}
    return id2label


# Save the mask as an image
def save_mask(mask, filename):
    visual_mask = (mask * 255).astype(np.uint8)
    image = Image.fromarray(visual_mask)
    image.save(os.path.join(output_dir, filename))


# Process a single example from the dataset
# Process a single example from the dataset
def process_example(dataset, id2label, example_id=1):
    example = dataset['train'][example_id]
    seg = np.array(example['annotation'])

    instance_seg = seg[:, :, 1]  # green channel encodes instances
    class_id_map = seg[:, :, 0]  # red channel encodes semantic category

    # Save the masks
    save_mask(instance_seg, f'instance_seg_{example_id}.png')
    save_mask(class_id_map, f'class_id_map_{example_id}.png')
    save_mask(seg, f'seg_{example_id}.png')

    # Separate masks for each instance
    unique_instances = np.unique(instance_seg)
    for instance_id in unique_instances:
        if instance_id != 0:  # Assuming 0 is the background
            object_mask = (instance_seg == instance_id)
            save_mask(object_mask, f'object_{instance_id}_mask_{example_id}.png')


def main():
    url = 'https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/instancesegmentation/instanceInfo100_train.txt'
    filename = wget.download(url)
    print(f'Downloaded {filename}')

    dataset = load_data()
    id2label = load_id_label_mapping()

    # You can process multiple examples by iterating through them
    # Here, just one example is processed
    process_example(dataset, id2label, example_id=1)


if __name__ == "__main__":
    main()
