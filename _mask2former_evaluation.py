import os
import random
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from dataset_preprocess.mask2former_mask.dataset_path import ID2LABEL

# Define the path to your images and model
image_dir = r"C:\Jinyoon Projects\datasets\combined_dataset_mask\test\images"
model_path = r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\mask_former_segment\train1-epoch20.pt"

# Load image file names and randomly select 10
image_files = os.listdir(image_dir)
selected_images = random.sample(image_files, 10)

# Increment each key by 1
id2label = {class_id + 1: class_name for class_id, class_name in ID2LABEL.items()}

# Initialize model with pre-trained weights
model = MaskFormerForInstanceSegmentation.from_pretrained(
    "facebook/maskformer-swin-base-ade",
    id2label=ID2LABEL,
    ignore_mismatched_sizes=True
)

# Load your custom trained weights into the model
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)  # strict=False allows partial loading

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

processor = MaskFormerImageProcessor()  # Initialize the processor used in your setup


# Function to get the mask
def get_mask(segmentation, segment_id):
    mask = (segmentation.cpu().numpy() == segment_id)
    visual_mask = (mask * 255).astype(np.uint8)
    return visual_mask


# Process each selected image
for image_file in selected_images:
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-processing
    results = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    # Display the original image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Original Image: {image_file}")

    # Find the segment with the highest score
    best_segment = max(results['segments_info'], key=lambda x: x['score'], default=None)

    if best_segment is not None:
        # Collect all segments with the same label_id as the best segment
        similar_segments = [seg for seg in results['segments_info'] if seg['label_id'] == best_segment['label_id']]

        # Visualizing and printing the best and similar segments
        for segment in similar_segments:
            if hasattr(model.config, 'id2label'):
                label_name = model.config.id2label.get(segment['label_id'], str(segment['label_id']))
            else:
                label_name = str(segment['label_id'])

            print(
                f"Image: {image_file} - Visualizing mask for instance: {label_name} (Confidence: {segment['score']:.2f})")
            mask = get_mask(results['segmentation'], segment['id'])

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.title(f"Mask: {label_name} ({segment['score']:.2f})")
    else:
        # No detection found
        plt.subplot(1, 2, 2)
        plt.imshow(np.zeros_like(np.array(image)))
        plt.axis('off')
        plt.title("No detection")
        print(f"Image: {image_file} - No detection")

    plt.show()
