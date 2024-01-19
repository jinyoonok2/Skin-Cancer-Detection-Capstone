import torch
import json
from pathlib import Path
from PIL import Image

from transformers import DetrFeatureExtractor
import numpy as np
from torch.utils.data import DataLoader

from dataset_preprocess.detr_coco_json.coco_dataset_path import *

# we reduce the size and max_size to be able to fit the batches in GPU memory
FEATURE_EXTRACTOR = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic", size=500, max_size=600)


class CocoPanoptic(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, ann_file, feature_extractor):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])

        # Optional: Check if each image has at least one annotation
        image_ids_with_annotations = {ann['image_id'] for ann in self.coco['annotations']}
        assert all(img['id'] in image_ids_with_annotations for img in self.coco['images'])

        self.img_folder = img_folder
        self.ann_folder = Path(ann_folder)
        self.ann_file = ann_file
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        img_info = self.coco['images'][idx]
        image_id = img_info['id']
        img_path = Path(self.img_folder) / img_info['file_name']

        img = Image.open(img_path).convert('RGB')

        # Find all annotations (segments) for this image
        segments_info = [ann for ann in self.coco['annotations'] if ann['image_id'] == image_id]

        # Reformulate annotations for feature extractor
        formatted_annotations = {
            "image_id": image_id,
            "segments_info": segments_info,
            "file_name": img_info['file_name']
        }

        # Preprocess image and target
        encoding = self.feature_extractor(images=img, annotations=formatted_annotations, masks_path=self.ann_folder,
                                          return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

    def __len__(self):
        return len(self.coco['images'])

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoded_input = FEATURE_EXTRACTOR.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoded_input['pixel_values']
    batch['pixel_mask'] = encoded_input['pixel_mask']
    batch['labels'] = labels
    return batch

TRAIN_DATASET = CocoPanoptic(img_folder=COCO_TRAIN_PATH,
                             ann_folder= os.path.join(COCO_TRAIN_PATH),
                             ann_file= COCO_TRAIN_ANNOTATION_PATH,
                             feature_extractor=FEATURE_EXTRACTOR)

VALID_DATASET = CocoPanoptic(img_folder=COCO_VALID_PATH,
                             ann_folder= os.path.join(COCO_VALID_PATH),
                             ann_file= COCO_VALID_ANNOTATION_PATH,
                             feature_extractor=FEATURE_EXTRACTOR)

TEST_DATASET = CocoPanoptic(img_folder=COCO_TEST_PATH,
                            ann_folder= os.path.join(COCO_TEST_PATH),
                            ann_file= COCO_TEST_ANNOTATION_PATH,
                            feature_extractor=FEATURE_EXTRACTOR)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, collate_fn=collate_fn, batch_size=3, shuffle=True)
VALID_DATALOADER = DataLoader(VALID_DATASET, collate_fn=collate_fn, batch_size=1)
TEST_DATALOADER = DataLoader(TEST_DATASET, collate_fn=collate_fn, batch_size=1)

if __name__ == '__main__':
    pixel_values, target = TRAIN_DATASET[2]
    print(pixel_values.shape)
    print(target.keys())

    print("Number of training examples:", len(TRAIN_DATASET))
    print("Number of validation examples:", len(VALID_DATASET))
    print("Number of test examples:", len(TEST_DATASET))