from transformers import AutoModelForUniversalSegmentation
import torch
from PIL import Image
from one_former_custom_dataload import test_dataloader, processor,id2label  # Assuming these are needed for processing
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from dataset_preprocess.one_former.dataset_path import TEST_PATH_IMAGES, TEST_PATH_MASKS

id2label_eval = {**id2label, 0: "background"}


def resize_mask(segmentation_map, output_size=(256, 256)):
    # # Convert the tensor to a numpy array for easy manipulation
    mask_np = segmentation_map.cpu().numpy().astype(np.uint8)

    mask_image = Image.fromarray(mask_np)
    resized_mask = mask_image.resize(output_size, Image.NEAREST)  # Use NEAREST to avoid introducing new classes
    return resized_mask

def process_and_save_mask(image_path, mask_output_path, model, processor, device):
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Prepare the image for the model
    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)

    # Forward pass to get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess to get the semantic segmentation
    semantic_segmentation = processor.post_process_semantic_segmentation(outputs)[0]

    # Resize the segmentation map to desired output size and print unique IDs
    resized_mask = resize_mask(semantic_segmentation, output_size=(256, 256))

    # Save the resized mask
    resized_mask.save(mask_output_path)

    print(f"Saved mask for {os.path.basename(image_path)} at {mask_output_path}")

if __name__ == '__main__':
    # Initialize your pretrained model
    latest_checkpoint_path = r"C:\Jinyoon Projects\0_Skin-Cancer-Detection-Capstone\runs\one_former_segment\checkpoint_epoch_16.pt"  # Update with your actual path
    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)

    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.model.is_training = False

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define paths
    test_images = os.listdir(TEST_PATH_IMAGES)[:5]  # Adjust this to control the number of images processed
    mask_output_dir = 'path_to_save_masks'  # Update with your actual path
    os.makedirs(mask_output_dir, exist_ok=True)

    # Process images and save masks
    for img_name in test_images:
        img_path = os.path.join(TEST_PATH_IMAGES, img_name)
        mask_output_path = os.path.join(mask_output_dir, os.path.splitext(img_name)[0] + '_mask.png')
        if os.path.isfile(img_path):
            process_and_save_mask(img_path, mask_output_path, model, processor, device)