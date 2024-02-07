from _mask2former_custom_dataload_semantic import weighted_train_dataloader
from transformers import Mask2FormerForUniversalSegmentation
import torch
from tqdm.auto import tqdm
from pathlib import Path
import os

if __name__ == '__main__':
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-instance")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Directory where you want to save and load your model checkpoints
    checkpoint_dir = 'mask2former/train1-epoch50'
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if it does not exist


    # Function to find the latest checkpoint in the directory
    def find_latest_checkpoint(checkpoint_dir):
        checkpoint_paths = [p for p in Path(checkpoint_dir).glob("checkpoint_epoch_*.pt")]
        if checkpoint_paths:
            latest_checkpoint = max(checkpoint_paths, key=os.path.getctime)
            return latest_checkpoint
        return None


    # Try to load the latest checkpoint
    latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    start_epoch = 0

    if latest_checkpoint_path and os.path.isfile(latest_checkpoint_path):
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("No checkpoint found, starting from scratch.")

    # Training loop
    num_epochs = 50
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        num_samples = 0
        model.train()
        for idx, batch in enumerate(tqdm(weighted_train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item() * batch_size  # Scale loss by batch size
            num_samples += batch_size

            # Optimization
            optimizer.step()

        # Average loss for the epoch
        avg_loss = running_loss / num_samples
        print(f"Epoch: {epoch}, Average Loss: {avg_loss}")

        # Save model and optimizer states after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,  # Save the average loss for the epoch
        }, checkpoint_path)

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)






