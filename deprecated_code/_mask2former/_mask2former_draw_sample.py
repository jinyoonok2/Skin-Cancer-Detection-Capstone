import os
import random
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import Mask2FormerForUniversalSegmentation, MaskFormerImageProcessor
from dataset_preprocess.mask2former_mask.dataset_path import ID2LABEL
from _mask2former_custom_dataload_semantic import test_dataloader, preprocessor, palette, id2label
from transformers import AutoImageProcessor
from matplotlib import colormaps as mcm

from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


# Normalize the palette to 0-1 range for matplotlib
palette_normalized = palette / 255.0
cmap = ListedColormap(palette_normalized)


# Function to create legend patches for the existing labels in the mask
def create_legend_for_mask(mask, id2label, cmap):
    unique_labels = np.unique(mask)
    print(f"Unique labels in mask: {unique_labels}")  # Debugging line
    print("Labels and their corresponding names:")
    for label in unique_labels:
        if label in id2label:
            print(f"Label {label}: {id2label[label]}")  # Debugging line
    patches = [mpatches.Patch(color=cmap(labels), label=id2label[labels])
               for labels in unique_labels if labels in id2label]
    return patches
# Function to convert segmentation map to a color map
def convert_to_color_segmentation_map(segmentation_map, cmap):
    segmentation_map = np.array(segmentation_map, dtype=int)  # Ensure it's an integer array
    color_segmentation_map = cmap(segmentation_map)
    return color_segmentation_map

# Function to save the visualizations of predictions with legends
def save_predictions(batch, predicted_segmentation_maps, id2label, cmap, save_dir):
    num_images = len(batch['original_images'])

    for i in range(num_images):
        # Extract data from the batch
        original_image = batch['original_images'][i]
        original_mask = batch['mask_labels'][i].cpu().numpy().squeeze()
        predicted_mask = predicted_segmentation_maps[i].cpu().numpy()
        image_name = batch['image_names'][i]

        # Define file paths for saving
        original_image_path = os.path.join(save_dir, f"{image_name}")
        actual_mask_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_actual.png")
        predicted_mask_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_predicted.png")

        # Save Original Image
        plt.imsave(original_image_path, original_image)

        # Create and Save Actual Mask with legend
        fig, ax = plt.subplots()
        actual_mask_color = convert_to_color_segmentation_map(original_mask, cmap)
        ax.imshow(actual_mask_color)
        legend_patches = create_legend_for_mask(original_mask, id2label, cmap)
        ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_patches))
        plt.axis('off')
        plt.savefig(actual_mask_path, bbox_inches='tight')
        plt.close(fig)

        # Create and Save Predicted Mask with legend
        fig, ax = plt.subplots()
        predicted_mask_color = convert_to_color_segmentation_map(predicted_mask, cmap)
        ax.imshow(predicted_mask_color)
        legend_patches = create_legend_for_mask(predicted_mask, id2label, cmap)
        ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_patches))
        plt.axis('off')
        plt.savefig(predicted_mask_path, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved images and masks for {image_name}")

if __name__ == '__main__':
    # Define the path to your specific checkpoint
    latest_checkpoint_path = r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\mask_former_segment\checkpoint_epoch_4_old.pt"  # Update this path

    # Prepare the model
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance",
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    # Load the checkpoint if it exists
    if os.path.isfile(latest_checkpoint_path):
        print(f"Loading model from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Checkpoint file not found, make sure the path is correct.")

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Define the save directory
    save_dir = 'mask2former_results_comparison'  # Update this path to your desired directory
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over the entire test dataset
    for batch in test_dataloader:
        # Perform a forward pass to get predictions
        with torch.no_grad():
            outputs = model(batch["pixel_values"].to(device))

        # Post-process model outputs to get predicted segmentation maps
        target_sizes = [(image.shape[0], image.shape[1]) for image in batch["original_images"]]
        predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                      target_sizes=target_sizes)

        # Save visualizations to files
        save_predictions(batch, predicted_segmentation_maps, ID2LABEL, cmap, save_dir)