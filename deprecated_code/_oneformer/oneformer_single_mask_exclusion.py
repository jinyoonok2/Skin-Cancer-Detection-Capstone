from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
from dataset_preprocess.one_former.dataset_path import COMB_TRAIN_PATH_MASKS, COMB_TRAIN_PATH_IMAGES, ID2LABEL, TEST_PATH_MASKS, TEST_PATH_IMAGES
import matplotlib.pyplot as plt
import albumentations as A

from shutil import copy2
import os

from torch.utils.data import WeightedRandomSampler

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# Increment each key by 1
id2label = {class_id + 1: class_name for class_id, class_name in ID2LABEL.items()}

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, transform=None):
        self.processor = processor
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([img for img in os.listdir(image_dir) if img.endswith('.jpg')])
        self.masks = sorted([mask for mask in os.listdir(mask_dir) if mask.endswith('_mask.png')])
        self.transform = transform
    def __getitem__(self, idx):
        # Load image and mask

        # Load image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
        mask = Image.open(mask_path)
        mask = np.array(mask)  # Convert mask to numpy array

        # Apply transformations to the image and mask
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=mask)  # Convert PIL image to numpy array
            image = augmented['image']
            mask = augmented['mask']

        # Use processor to convert this to a list of binary masks, labels, text inputs, and task inputs
        inputs = self.processor(images=image, segmentation_maps=mask, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return inputs

    def __len__(self):
        return len(self.images)

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

    def get_mask_channels(self, idx):
        """
        Get the number of channels in the mask_labels of a specific image.
        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            int: Number of channels in mask_labels.
        """
        data = self[idx]  # Use __getitem__ to process the image and mask
        mask_labels = data['mask_labels']
        return mask_labels.shape[0]

def create_train_transform():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        A.Rotate(limit=35, p=0.5),  # Random rotation between -35 and +35 degrees, 50% chance to apply rotation
    ])
    return train_transform

processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)
processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx
processor.image_processor.do_resize = False

train_dataset = CustomDataset(COMB_TRAIN_PATH_IMAGES, COMB_TRAIN_PATH_MASKS, processor)
test_dataset = CustomDataset(TEST_PATH_IMAGES, TEST_PATH_MASKS, processor)

if __name__ == '__main__':
    example = train_dataset[0]
    for k, v in example.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    print(processor.tokenizer.batch_decode(example["text_inputs"]))

    single_channel_dir = 'single_channel'  # Path to the directory for single channel masks and images
    multi_channel_dir = 'multi_channel'  # Path to the directory for multi-channel masks and images

    os.makedirs(os.path.join(single_channel_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(single_channel_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(multi_channel_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(multi_channel_dir, 'masks'), exist_ok=True)

    for idx in range(len(train_dataset)):
        image_path = train_dataset.images[idx]
        mask_path = train_dataset.masks[idx]
        num_channels = train_dataset.get_mask_channels(idx)  # Use the new method

        if num_channels == 1:
            print(f"Single channel mask found: {mask_path}")
            copy2(os.path.join(train_dataset.image_dir, image_path),
                  os.path.join(single_channel_dir, 'images', image_path))
            copy2(os.path.join(train_dataset.mask_dir, mask_path), os.path.join(single_channel_dir, 'masks', mask_path))
        else:
            copy2(os.path.join(train_dataset.image_dir, image_path),
                  os.path.join(multi_channel_dir, 'images', image_path))
            copy2(os.path.join(train_dataset.mask_dir, mask_path), os.path.join(multi_channel_dir, 'masks', mask_path))