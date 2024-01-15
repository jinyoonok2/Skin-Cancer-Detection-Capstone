import torch
from transformers import DetrForSegmentation
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from _detr_custom_dataloader import CHECKPOINT, TRAIN_DATALOADER, VAL_DATALOADER, TRAIN_DATASET

# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        categories = TRAIN_DATASET.coco.cats
        self.id2label = {k: v['name'] for k, v in categories.items()}
        self.model = DetrForSegmentation.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(self.id2label),
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here:
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER

def main():
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    batch = next(iter(TRAIN_DATALOADER))
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
    print(outputs.logits.shape)

    # settings
    MAX_EPOCHS = 50

    # pytorch_lightning < 2.0.0
    # trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

    # pytorch_lightning >= 2.0.0
    trainer = Trainer(devices=0, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1,
                      accumulate_grad_batches=8, log_every_n_steps=5)

    trainer.fit(model)


if __name__ == '__main__':
    main()
