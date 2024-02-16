
from datasets import load_dataset, DatasetDict
from datasets import load_metric
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from transformers import AutoImageProcessor
import evaluate

from torchvision.transforms.functional import to_pil_image
import torch


model_checkpoint = "microsoft/swinv2-tiny-patch4-window8-256" # pre-trained model from which to fine-tune
batch_size = 32 # batch size for training and evaluation

# Load the training and validation datasets individually
train_dataset = load_dataset(r"C:\Jinyoon Projects\datasets\combined_dataset_swin_split", data_dir="train", trust_remote_code=True)['train']
valid_dataset = load_dataset(r"C:\Jinyoon Projects\datasets\combined_dataset_swin_split", data_dir="valid", trust_remote_code=True)['train']  # Assuming 'train' is not a typo and refers to the structure of the loaded dataset

# Combine the loaded datasets into a single DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'valid': valid_dataset
})

print(dataset)

# Load the accuracy metric
metric = evaluate.load('accuracy')

# Assuming the 'labels' attribute correctly exists in your dataset's features
# If 'labels' is not the correct key, adjust it to match your dataset structure
labels = dataset['train'].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# 4. Image processor
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)


# 5. customize dataset for train/valid
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )


def preprocess_train(example_batch):
    """Print keys of an example and apply train_transforms across a batch."""
    # Print keys of the first element in the batch to understand its structure
    print(f"Keys in the batch: {list(example_batch.keys())}")

    # Check if 'image' key exists in the batch, and print the first item's keys if 'image' is not found
    if 'image' not in example_batch:
        first_item_keys = list(example_batch[list(example_batch.keys())[0]][0].keys()) if example_batch[
            list(example_batch.keys())[0]] else 'Empty Batch'
        print(f"No 'image' key found. First item keys (if available): {first_item_keys}")
        return example_batch

    # Continue with the original processing if 'image' key exists
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# 6. split train/valid(test)
train_ds = dataset['train']
val_ds = dataset['valid']
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

if __name__ == '__main__':
    print(dataset["valid"])
    example = dataset["valid"][10]
    print(example)
    print(dataset["valid"].features)

    # 1. Plot image
    image = example['image']

    # If example['image'] is already an image array, you can directly use:
    # image = example['image']
    plt.imshow(image)
    plt.axis('off')  # No axes for images
    plt.show()

    # 2. Check label
    print("Example label: ", example['label'])

    print("id2label 2: ", id2label[0])

    print("image processor: ", image_processor)

    print("Train dataset idx 0: ", train_ds[0])

    if "height" in image_processor.size:
        print(image_processor.size["height"])
        print(image_processor.size["width"])

    # Assuming `example` contains a single image example from your dataset
    image = example['image']  # This should be a PIL image

    # Wrap the image in a list to mimic a batch
    example_batch = {'image': [image]}

    # Apply the transform (choose preprocess_train or preprocess_val as needed)
    processed_batch = preprocess_train(example_batch)  # or preprocess_val for validation images

    # Extract the processed image tensor
    processed_image_tensor = processed_batch['pixel_values'][0]

    # Convert the tensor to a PIL image for visualization
    # Note: This step requires the image tensor to be in CPU memory
    processed_image = to_pil_image(processed_image_tensor)

    # Visualize the processed image
    plt.imshow(processed_image)
    plt.axis('off')  # No axes for images
    plt.show()




