from _swinv2_custom_dataload import (id2label, label2id, batch_size, model_checkpoint,
                                     metric, train_ds, val_ds, image_processor, labels, dataset)
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
import torch

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Each batch consists of 2 keys, namely pixel_values and labels.
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def show_images(images, labels, n_max=5):
    fig, axs = plt.subplots(1, n_max, figsize=(15, 3))
    for i, (img, label) in enumerate(zip(images[:n_max], labels[:n_max])):
        axs[i].imshow(img.permute(1, 2, 0))
        axs[i].set_title(f'Label: {id2label[label]}')
        axs[i].axis('off')
    plt.show()

if __name__ == '__main__':
    # train the model
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

    model_name = model_checkpoint.split("/")[-1]

    # define the training configuration and the evaluation metric.
    # remove_unused_columns=False. This one will drop any features not used by the model's call function.
    args = TrainingArguments(
        output_dir= f"{model_name}-finetuned-cd",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # trainer = CustomTrainer(
    #     model,
    #     args,
    #     train_dataset=train_ds,
    #     eval_dataset=val_ds,
    #     compute_metrics=compute_metrics,
    #     data_collator=collate_fn,
    # )

    trainer.print_dataset_sample()

    # Assuming `custom_trainer` is an instance of your CustomTrainer
    train_dataloader = trainer.get_train_dataloader()

    # Get the first batch
    for batch in train_dataloader:
        images, labels = batch["pixel_values"], batch["labels"]
        show_images(images, labels)
        break  # Only show the first batch

    train_results = trainer.train()
    # rest is optional but nice to have
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()

    # some nice to haves:
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


