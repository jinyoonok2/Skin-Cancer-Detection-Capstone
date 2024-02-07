from transformers import AutoModelForUniversalSegmentation
import torch
from PIL import Image
import requests
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from _one_former_custom_dataload import test_dataloader, processor  # Custom dataloader and processor


if __name__ == '__main__':
    # Initialize your pretrained model
    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)

    # Path to your saved .pt file
    model_path = 'runs/one_former_segment/checkpoint_epoch_2.pt'

    # Load the checkpoint
    checkpoint = torch.load(model_path)

    # Load the weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    model.model.is_training = False

    # Set the model to evaluation mode
    model.eval()

    # Load image from local directory
    image_path = r"C:\Jinyoon Projects\datasets\combined_dataset_semantic\train_valid_combined_SE\images\ISIC_0000000_MNV.jpg" # Replace with your local image path
    image = Image.open(image_path)

    # Prepare image for the model
    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")

    # Print input shapes
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    # Forward pass (no need for gradients at inference time)
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocessing
    semantic_segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]


    def draw_semantic_segmentation(segmentation):
        # Get the used color map
        viridis = cm.get_cmap('viridis', torch.max(segmentation))
        # Get all the unique numbers
        labels_ids = torch.unique(segmentation).tolist()
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        handles = []
        for label_id in labels_ids:
            label = model.config.id2label[label_id]
            color = viridis(label_id)
            handles.append(mpatches.Patch(color=color, label=label))
        ax.legend(handles=handles)


    draw_semantic_segmentation(semantic_segmentation)
    plt.show()  # Display the plot

