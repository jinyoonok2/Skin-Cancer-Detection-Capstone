import os
import torchvision
from torchvision.transforms import functional as F
import torch
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader

# settings
CHECKPOINT = 'facebook/detr-resnet-50'
ANNOTATION_FILE_NAME = "_annotations.coco.json"
DATASET_DIR = 'C:\Jinyoon Projects\datasets'
TRAIN_DIRECTORY = os.path.join(DATASET_DIR, r'combined_dataset_coco\train')
VAL_DIRECTORY = os.path.join(DATASET_DIR, r'combined_dataset_coco\valid')
TEST_DIRECTORY = os.path.join(DATASET_DIR, r'combined_dataset_coco\test')

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

# CocoDetection class as you defined
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible
    # to directly batch together images. Hence they pad the images to the biggest
    # resolution in a given batch, and create a corresponding binary pixel_mask
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

# Instantiate the datasets
TRAIN_DATASET = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY,
    image_processor=image_processor,
    train=True)
VAL_DATASET = CocoDetection(
    image_directory_path=VAL_DIRECTORY,
    image_processor=image_processor,
    train=False)
TEST_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY,
    image_processor=image_processor,
    train=False)

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)

if __name__ == "__main__":
    # Print the number of examples in each dataset
    print("Number of training examples:", len(TRAIN_DATASET))
    print("Number of validation examples:", len(VAL_DATASET))
    print("Number of test examples:", len(TEST_DATASET))