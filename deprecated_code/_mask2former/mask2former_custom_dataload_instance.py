import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, RandomSampler
import albumentations as A
from transformers import MaskFormerImageProcessor
from dataset_preprocess.mask2former_mask.dataset_path import (COMB_TRAIN_PATH_IMAGES, COMB_TRAIN_PATH_MASKS, ID2LABEL)
import yaml

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

magnification_factors = {
    'MEL': 4, 'MNV': 1, 'NV': 1, 'BCC': 2, 'AK': 8,
    'BKL': 4, 'DF': 8, 'VASC': 2, 'SCC': 8
}

# Increment each key by 1
id2label = {class_id + 1: class_name for class_id, class_name in ID2LABEL.items()}

class CustomImageSegmentationDataset(Dataset):
    """Image segmentation dataset for custom directory structure."""

    def __init__(self, image_dir, mask_dir, processor, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            processor: Processor to prepare inputs for model.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transform = transform
        self.images = sorted([img for img in os.listdir(image_dir) if img.endswith('.jpg')])
        self.masks = sorted([mask for mask in os.listdir(mask_dir) if mask.endswith('_mask.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = np.array(Image.open(image_path).convert("RGB"))
        seg = np.array(Image.open(mask_path))

        instance_seg = seg[:, :, 1]  # Green channel for instance segmentation
        class_id_map = seg[:, :, 0]  # Red channel for class segmentation
        class_labels = np.unique(class_id_map)

        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
            # image = image.transpose(2, 0, 1)

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # Some image does not have annotation (all ignored)
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
            inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return inputs

    def get_class_id_for_image(self, idx):
        # Implement logic to determine the class ID for the image at the given index
        # For example, you might read the class ID from the mask file
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = np.array(Image.open(mask_path))
        class_id = np.max(mask[:, :, 0])  # Example: extract class ID from the red channel of the mask
        return class_id

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

def create_train_transform():
    train_transform = A.Compose([
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),  # 50% chance to apply horizontal flip
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=35, p=0.5),  # Random rotation between -35 and +35 degrees, 50% chance to apply rotation
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),  # Apply only to 'image'
    ])
    return train_transform


def get_data_loader(train_dataset, magnification_factors, id2label):
    # Calculate sample weights based on magnification factors
    sample_weights = []

    for idx in range(len(train_dataset)):
        class_id = train_dataset.get_class_id_for_image(idx)
        class_name = id2label[class_id]  # Map class_id to class name
        weight = magnification_factors[class_name]
        sample_weights.append(weight)

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create DataLoader with the WeightedRandomSampler
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, sampler=sampler)

    return train_dataloader

# Define a function to unnormalize and display images
def visualize_data_and_batch(data, dataloader, index):
    """
    Visualizes the images and masks from the dataset and the batch from the DataLoader.

    Args:
        data (CustomImageSegmentationDataset): The dataset object.
        dataloader (DataLoader): The DataLoader object.
        index (int): Index of the sample in the dataset and batch to visualize.
    """
    # Visualize data from dataset
    inputs = data[index]
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"Dataset - {k}: {v.shape}")

    # Visualize data from batch of the DataLoader
    batch = next(iter(dataloader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"Batch - {k}: {v.shape}")
        else:
            print(f"Batch - {k}: {len(v)}")

    # Assuming you already have the 'batch' from your dataloader
    unnormalized_image = (batch["pixel_values"][index].numpy() * ADE_STD[:, None, None]) + ADE_MEAN[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    image_pil = Image.fromarray(unnormalized_image)

    # Fetch the class labels and masks
    class_labels = batch["class_labels"][index]
    mask_labels = batch["mask_labels"][index]

    # Visualization
    fig, axes = plt.subplots(1, 1 + len(class_labels), figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Plot each mask
    mask_plotted = 0
    for i, class_label in enumerate(class_labels):
        if class_label.item() == 0:
            # Skip background class if it doesn't have an explicit mask
            continue

        # Proceed if it's an actual class
        visual_mask = (mask_labels[mask_plotted].bool().numpy() * 255).astype(np.uint8)
        label_description = id2label[class_label.item()]

        axes[i + 1].imshow(visual_mask, cmap='gray')
        axes[i + 1].set_title(f"Mask for: {label_description}")
        axes[i + 1].axis('off')
        mask_plotted += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # note that you can include more fancy data augmentation methods here
    train_transform = create_train_transform()

    processor = MaskFormerImageProcessor(do_reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False,
                                         do_normalize=False)

    train_dataset = CustomImageSegmentationDataset(image_dir=COMB_TRAIN_PATH_IMAGES, mask_dir=COMB_TRAIN_PATH_MASKS, processor=processor,
                                                   transform=train_transform)

    # Use the function
    train_dataloader = get_data_loader(train_dataset, magnification_factors, id2label)

    # Visualize
    batch_index = 1
    visualize_data_and_batch(train_dataset, train_dataloader, batch_index)





