import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
from _detr_custom_dataloader import *
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
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
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
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
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub("jinyoonok/jinyoon-skin",
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub("jinyoonok/jinyoon-skin",
                                        commit_message=f"Training done")
        pl_module.model.push_to_hub("jinyoonok/jinyoon-skin",
                                    commit_message=f"Training done")


if __name__ == '__main__':
    # PyTorch provides the function torch.set_float32_matmul_precision to optimize the use of Tensor Cores
    torch.set_float32_matmul_precision('medium')  # or 'high'

    batch = next(iter(train_dataloader))
    print(batch.keys())

    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
    print(outputs.logits.shape)

    wandb_logger = WandbLogger(project="DETR_ISIC", name="detr_isic")

    early_stop_callback = EarlyStopping(
        monitor="validation_loss",  # Adjusted to monitor validation_loss
        patience=3,
        verbose=False,
        mode="min"
    )

    # Setup model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{validation_loss:.2f}",  # Adjusted filename template
        save_top_k=3,
        monitor="validation_loss",  # Adjusted to monitor validation_loss
        mode="min",
        every_n_epochs=1
    )

    # trainer settings
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=4,
        max_steps=-1,
        # val_check_interval=1,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        precision=16,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        callbacks=[PushToHubCallback(), early_stop_callback, checkpoint_callback],
    )

    # # resume settings
    # trainer = Trainer(
    #     resume_from_checkpoint="checkpoints/epoch=0-validation_loss=0.91.ckpt",
    #     accelerator="gpu",
    #     devices=1,
    #     max_epochs=4,
    #     max_steps=-1,
    #     # val_check_interval=1,
    #     check_val_every_n_epoch=1,
    #     gradient_clip_val=1.0,
    #     precision=16,
    #     logger=wandb_logger,
    #     num_sanity_val_steps=0,
    #     callbacks=[PushToHubCallback(), early_stop_callback, checkpoint_callback],
    # )

    # run training
    trainer.fit(model)

# huggingface-cli login
# wandb login

# config = {"max_epochs":30,
#           "val_check_interval":0.2, # how many times we want to validate during an epoch
#           "check_val_every_n_epoch":1,
#           "gradient_clip_val":1.0,
#           "num_training_samples_per_epoch": 800,
#           "lr":3e-5,
#           "train_batch_sizes": [8],
#           "val_batch_sizes": [1],
#           # "seed":2022,
#           "num_nodes": 1,
#           "warmup_steps": 300, # 800/8*30/10, 10%
#           "result_path": "./result",
#           "verbose": True,
#           }