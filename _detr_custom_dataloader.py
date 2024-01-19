from dataset_preprocess.detr_coco_json.coco_dataset_path import *
import torchvision
import torch
from transformers import DetrImageProcessor
import numpy as np
import os
from torch.utils.data import DataLoader

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "annotations.coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

train_dataset = CocoDetection(img_folder=COCO_TRAIN_PATH, processor=processor)
val_dataset = CocoDetection(img_folder=COCO_VALID_PATH, processor=processor, train=False)
cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True, persistent_workers=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2, persistent_workers=True, num_workers=2)

if __name__ == '__main__':
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    pixel_values, target = train_dataset[0]
    print(pixel_values.shape)
    print(target)

