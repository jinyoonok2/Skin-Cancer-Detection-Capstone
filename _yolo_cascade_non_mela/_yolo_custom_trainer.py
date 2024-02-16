from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import torch_distributed_zero_first, de_parallel
from _yolo_custom_dataset import *
from dataset_preprocess.yolo_txt_cascade_non_mela._train_magnification import magnification_factors

class_augment_check = {i: magnification_factors[class_name] > 1 for i, class_name in enumerate(magnification_factors)}

# inherit from Detection Trainer which inherits from BaseTrainer
# DetectionTrainer: ultralytics/models/yolo/detect.py
# BaseTrainer: ultralytics/engine/trainer.py
class CustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_custom_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

# Rewritten function from build_yolo_dataset
# ultralytics/data/build.py
def build_custom_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset."""
    return CustomYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        class_augment_check=class_augment_check
    )