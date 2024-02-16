import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from transformers import Mask2FormerImageProcessor
from dataset_preprocess.mask2former_mask.dataset_path import (COMB_TRAIN_PATH_IMAGES, COMB_TRAIN_PATH_MASKS, TEST_PATH_IMAGES, TEST_PATH_MASKS, ID2LABEL)

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

# Create a preprocessor
preprocessor = Mask2FormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False,
                                        do_normalize=False)

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

magnification_factors = {
    'MEL': 4, 'MNV': 1, 'NV': 1, 'BCC': 2, 'AK': 8,
    'BKL': 4, 'DF': 8, 'VASC': 2, 'SCC': 8
}

# Increment each key by 1
id2label = {class_id + 1: class_name for class_id, class_name in ID2LABEL.items()}

# Creating a color palette
palette = np.array([
    [0, 0, 0],        # Background - Black
    [128, 0, 0],      # Class 1 - Maroon
    [0, 128, 0],      # Class 2 - Green
    [128, 128, 0],    # Class 3 - Olive
    [0, 0, 128],      # Class 4 - Navy
    [128, 0, 128],    # Class 5 - Purple
    [0, 128, 128],    # Class 6 - Teal
    [128, 128, 128],  # Class 7 - Silver
    [64, 0, 0],       # Class 8 - Dark Red
    [192, 192, 192]   # Class 9 - Light Grey
], dtype=np.uint8)

class CustomImageSegmentationDataset(Dataset):
    """Image segmentation dataset for custom directory structure."""

    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([img for img in os.listdir(image_dir) if img.endswith('.jpg')])
        self.masks = sorted([mask for mask in os.listdir(mask_dir) if mask.endswith('_mask.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image_name = self.images[idx]  # This gets the image name with .jpg extension

        image = Image.open(image_path).convert('RGB')
        segmentation_map = Image.open(mask_path).convert('L')  # Convert to grayscale because masks are typically in single channel

        # Convert to numpy arrays for the original (untransformed) images and masks
        original_image = np.array(image)
        original_segmentation_map = np.array(segmentation_map)

        # Apply transform
        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, segmentation_map = transformed['image'], transformed['mask']

        # Convert from HxWxC to CxHxW
        image = image.transpose(2, 0, 1)

        return image, segmentation_map, original_image, original_segmentation_map, image_name

    def get_class_id_for_image(self, idx):
        """
        Get the class ID for the mask of a specific image.
        Assumes that each mask contains only one class apart from the background.
        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            int: Class ID of the mask.
        """
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)

        # Use np.max to get the highest value in the mask, assuming it represents the class ID
        class_id = np.max(mask_array)
        return class_id

def create_train_transform():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        A.Rotate(limit=35, p=0.5),  # Random rotation between -35 and +35 degrees, 50% chance to apply rotation
    ])
    return train_transform

def create_test_transform():
    test_transform = A.Compose([
        A.Resize(width=256, height=256),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    return test_transform

def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]

    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]
    batch["image_names"] = inputs[4]

    return batch


def get_weighted_data_loader(train_dataset, magnification_factors, id2label, collate_fn):
    # Calculate weights for each sample in the dataset
    sample_weights = []
    for idx in range(len(train_dataset)):
        # Get class ID for the current sample
        class_id = train_dataset.get_class_id_for_image(idx)

        # Find the class name using id2label (class_id should match the keys in id2label)
        class_name = id2label[class_id]

        # Get the magnification factor for the class
        magnification_factor = magnification_factors.get(class_name,
                                                         1)  # Default to 1 if class_name not in magnification_factors

        # Append the weight
        sample_weights.append(magnification_factor)

    # Convert to a PyTorch tensor
    sample_weights = torch.tensor(sample_weights)

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create DataLoader with the WeightedRandomSampler
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn,
                                  sampler=sampler)

    return train_dataloader

# note that you can include more fancy data augmentation methods here
train_transform = create_train_transform()
train_dataset = CustomImageSegmentationDataset(image_dir=COMB_TRAIN_PATH_IMAGES, mask_dir=COMB_TRAIN_PATH_MASKS, transform=train_transform)
test_transform = create_test_transform()
test_dataset = CustomImageSegmentationDataset(image_dir=TEST_PATH_IMAGES, mask_dir=TEST_PATH_MASKS, transform=test_transform)

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

weighted_train_dataloader = get_weighted_data_loader(train_dataset=train_dataset, magnification_factors=magnification_factors, id2label=id2label, collate_fn=collate_fn)

if __name__ == '__main__':

    # 1. Dataset Verification
    # Get image, segmentation_map
    image, segmentation_map, _, _ = train_dataset[0]

    print("Image shape: ", image.shape)
    print("Segmentation map shape: ", segmentation_map.shape)

    # Convert image from tensor format to image format
    unnormalized_image = (image * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

    # Get the class label using np.max, assuming only one class per mask
    class_label = np.max(segmentation_map)
    print("Class label in segmentation map:", class_label)

    # Translate class_label to class name, considering id2label starts from 1
    class_name = id2label.get(class_label, "Background")
    print("Class name:", class_name)

    # Convert segmentation map to color
    color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)  # height, width, 3
    color = palette[class_label] if class_label > 0 else palette[0]  # Get color for the class label, use background color for 0
    color_segmentation_map[segmentation_map > 0] = color  # Apply class color to mask where mask_labels > 0

    # Create a subplot of 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the original image
    axes[0].imshow(unnormalized_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the segmentation map
    axes[1].imshow(color_segmentation_map)
    axes[1].set_title(f'Segmentation Map - Class: {class_name}')
    axes[1].axis('off')

    plt.show()


    # 2. Dataloader Verification
    batch = next(iter(weighted_train_dataloader))

    # Assuming the batch contains 'pixel_values', 'mask_labels', and 'class_labels'
    pixel_values = batch["pixel_values"][0].numpy()  # Get the first image in the batch
    mask_labels = batch["mask_labels"][0].numpy()  # Get the first mask in the batch

    # Unnormalize the image
    unnormalized_image = (pixel_values * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

    # Get label name for the first image in the batch
    label_name = id2label[batch["class_labels"][0].item()]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the unnormalized image
    axes[0].imshow(unnormalized_image)
    axes[0].set_title(f'Original Image - Label: {label_name}')
    axes[0].axis('off')

    # Display the mask
    # Assuming there's only one object class per mask, find the index of that class
    # If there are multiple classes, you might need to modify this part
    class_idx = np.max(mask_labels)
    visual_mask = (mask_labels == class_idx).astype(np.uint8) * 255  # Convert the mask to binary format
    # Assuming visual_mask has shape (1, height, width)
    visual_mask = np.squeeze(visual_mask)  # This removes the singleton dimension
    axes[1].imshow(visual_mask, cmap='gray')  # Using grayscale color map
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    plt.show()



