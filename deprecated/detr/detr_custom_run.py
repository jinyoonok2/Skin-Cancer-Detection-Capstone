import torch
from transformers import DetrForSegmentation, DetrConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from detr_custom_dataloader import TRAIN_DATALOADER, VALID_DATALOADER, TEST_DATALOADER

# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8


class DetrPanoptic(pl.LightningModule):

    def __init__(self, model, lr, lr_backbone, weight_decay):
        super().__init__()

        self.model = model

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

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
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VALID_DATALOADER

    def test_dataloader(self):
        return TEST_DATALOADER

def main():
    # 1. Here we load the model trained on COCO panoptic. We decide to only train the class labels classifier from scratch,
    # and further fine-tune the bounding box regressor and mask head.
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    state_dict = model.state_dict()
    # Remove class weights
    del state_dict["detr.class_labels_classifier.weight"]
    del state_dict["detr.class_labels_classifier.bias"]
    # define new model with custom class classifier
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50-panoptic", num_labels=250)
    model.load_state_dict(state_dict, strict=False)

    # 2. Next, we define the PyTorch LightningModule, and verify its outputs on a batch.
    model = DetrPanoptic(model=model, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    # pick the first training batch
    batch = next(iter(TRAIN_DATALOADER))
    # forward through the model
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

    print("Shape of logits:", outputs.logits.shape)
    print("Shape of predicted bounding boxes:", outputs.pred_boxes.shape)
    print("Shape of predicted masks:", outputs.pred_masks.shape)

    # 3. let's train! We train for a maximum of 25 epochs, and also use gradient clipping.
    trainer = Trainer(max_epochs=25, gradient_clip_val=0.1, accelerator="auto")
    trainer.fit(model)

if __name__ == '__main__':
    main()
