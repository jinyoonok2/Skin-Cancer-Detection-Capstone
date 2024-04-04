from transformers import AutoModelForUniversalSegmentation
import torch
from PIL import Image
import requests
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from one_former_custom_dataload import test_dataloader, processor, id2label  # Custom dataloader and processor
import os
from dataset_preprocess.one_former.dataset_path import TEST_PATH_MASKS, TEST_PATH_IMAGES
import numpy as np
from matplotlib.colors import ListedColormap

# Creating a color palette
palette = np.array([
    [0, 0, 0],        # Background - Black
    [128, 0, 0],      # Class 1 - Maroon
    [0, 128, 0],      # Class 2 - Green
    [128, 128, 0],    # Class 3 - Olive
    [0, 0, 128],      # Class 4 - Navy
    [128, 0, 128],    # Class 5 - Purple
    [0, 128, 128],    # Class 6 - Teal
    [128, 128, 128],  # Class 7 - Silver
    [64, 0, 0],       # Class 8 - Dark Red
    [192, 192, 192]   # Class 9 - Light Grey
], dtype=np.uint8)

id2label_eval = {**id2label, 0: "background"}

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
def process_and_save(image_path, mask_path, model, processor, device, cmap, id2label, save_dir):
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Prepare the image for the model
    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)

    # Forward pass to get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocessing to get the semantic segmentation
    predicted_segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    # Load actual mask
    actual_mask = np.array(Image.open(mask_path))

    # Convert segmentation maps to color maps
    predicted_mask_color = convert_to_color_segmentation_map(predicted_segmentation.cpu().numpy(), cmap)
    actual_mask_color = convert_to_color_segmentation_map(actual_mask, cmap)

    # Image name for saving
    image_name = os.path.basename(image_path)

    # Plot and save the original image
    plt.figure()
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{image_name}"), bbox_inches='tight')
    plt.close()

    # Adjusting these parameters for better legend spacing
    legend_offset = -0.15  # Increase this value if you need more space

    # Plot and save the actual mask with legend
    fig, ax = plt.subplots()
    ax.imshow(actual_mask_color)
    legend_patches = create_legend_for_mask(actual_mask, id2label, cmap)
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, legend_offset), ncol=3, frameon=False)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_actual.png"), bbox_inches='tight',
                pad_inches=1)
    plt.close(fig)

    # Plot and save the predicted mask with legend
    fig, ax = plt.subplots()
    ax.imshow(predicted_mask_color)
    legend_patches = create_legend_for_mask(predicted_segmentation.cpu().numpy(), id2label, cmap)
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, legend_offset), ncol=3, frameon=False)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_predicted.png"), bbox_inches='tight',
                pad_inches=1)
    plt.close(fig)

    print(f"Saved images and masks for {image_name}")

if __name__ == '__main__':
    # Initialize your pretrained model
    latest_checkpoint_path = r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\one_former_segment\checkpoint_epoch_16.pt"  # Update this path
    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)

    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.model.is_training = False

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Define the save directory
    save_dir = 'oneformer_results_comparison'  # Update this path to your desired directory
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over images in the test directory
    for img_name in os.listdir(TEST_PATH_IMAGES):
        img_path = os.path.join(TEST_PATH_IMAGES, img_name)
        mask_name = os.path.splitext(img_name)[0] + '_mask.png'  # Assuming mask file names match image names
        mask_path = os.path.join(TEST_PATH_MASKS, mask_name)

        if os.path.isfile(img_path) and os.path.isfile(mask_path):
            process_and_save(img_path, mask_path, model, processor, device, cmap, id2label_eval, save_dir)
