import os

COCO_DATASET_DIR = r'C:\Jinyoon Projects\datasets\combined_dataset_coco'
COCO_YAML_PATH = os.path.join(COCO_DATASET_DIR, 'data.yaml')
COCO_OLD_YAML_PATH = os.path.join(COCO_DATASET_DIR, 'data_old_class_10.yaml')

COCO_TRAIN_PATH = os.path.join(COCO_DATASET_DIR, 'train')
COCO_VALID_PATH = os.path.join(COCO_DATASET_DIR, 'valid')
COCO_TEST_PATH = os.path.join(COCO_DATASET_DIR, 'test')

COCO_TRAIN_ANNOTATION_PATH = os.path.join(COCO_TRAIN_PATH, 'annotations.coco.json')
COCO_VALID_ANNOTATION_PATH = os.path.join(COCO_VALID_PATH, 'annotations.coco.json')
COCO_TEST_ANNOTATION_PATH = os.path.join(COCO_TEST_PATH, 'annotations.coco.json')

COCO_TRAIN_MASK_PATH = os.path.join(COCO_TRAIN_PATH, 'masks')
COCO_VALID_MASK_PATH = os.path.join(COCO_VALID_PATH, 'masks')
COCO_TEST_MASK_PATH = os.path.join(COCO_TEST_PATH, 'masks')