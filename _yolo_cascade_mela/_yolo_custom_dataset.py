from ultralytics.data.augment import *
from ultralytics.data.dataset import *
from ultralytics.utils import DEFAULT_CFG, LOGGER

from torchvision.transforms.functional import to_pil_image
import os

# inherit from YOLODataset
# ultralytics/data/dataset.py
class CustomYOLODataset(YOLODataset):
    def __init__(self, *args, class_augment_check=None, task="detect", hyp=DEFAULT_CFG, **kwargs):
        """Initializes the CustomYOLODataset with a map of classes to their augmentation multipliers."""
        super().__init__(*args, **kwargs)
        self.class_augment_check = class_augment_check if class_augment_check else {}

        # Define separate transforms for each multiplier
        self.transforms_augment = self.build_transforms(hyp=hyp, augment=True)
        self.transforms_base = self.build_transforms(hyp=hyp, augment=False)

    def __getitem__(self, index):
        label_info = self.get_image_and_label(index)  # Get the original label info
        im_file = label_info['im_file']  # Extract image file path
        cls = int(label_info['cls'][0][0])

        augment_check = self.class_augment_check.get(cls, False)  # Default to False if class not found

        # Select the appropriate transforms based on the multiplier
        if augment_check:
            custom_transforms = self.transforms_augment
        else:
            custom_transforms = self.transforms  # Default transforms if no match
        # Apply the transformations
        augmented = custom_transforms(label_info)

        return augmented


    def build_transforms(self, hyp=None, augment=False):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms_alter(self, self.imgsz, hyp, augment=augment)  # Pass multipliers here
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms


# alter the transforms to apply class specific transforms
# hyp is chosen by default.yaml in ultralytics/cfg/default.yaml
# rewritten from v8_transforms function in ultralytics/data/augment.py
def v8_transforms_alter(dataset, imgsz, hyp, augment=False, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose([
        Mosaic(dataset, imgsz=imgsz, p=1.0),
        # Mosaic(dataset, imgsz=imgsz, p=1.0 if augment else 0.2), # the one creates that a lot of boxes
        # Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
        # CopyPaste(p=0.3 if augment else 0.0),  # default is 0.0 in default.yaml /call: hyp.copy_paste
        CopyPaste(p=hyp.copy_paste),
        RandomPerspective(
            degrees=45.0 if augment else 0.0,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
        )])
    flip_idx = dataset.data.get('flip_idx', [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    return Compose([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=0.5 if augment else 0.0),
        RandomFlip(direction='horizontal', p=0.5 if augment else 0.0, flip_idx=flip_idx)
    ])  # transforms