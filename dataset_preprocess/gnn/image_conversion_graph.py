import os
import cv2
import networkx as nx
import numpy as np


# Path to the directory containing class folders
data_dir = "path/to/your/dataset"

# List all class directories
class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Load images, assuming each class directory contains relevant images
images = {}  # Dictionary to hold images, keyed by class
for class_dir in class_dirs:
    class_dir_path = os.path.join(data_dir, class_dir)
    images[class_dir] = [cv2.imread(os.path.join(class_dir_path, img)) for img in os.listdir(class_dir_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

def segment_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def image_to_graph(segmented_image):
    # Find connected components
    num_labels, labels_im = cv2.connectedComponents(segmented_image)

    G = nx.Graph()
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Add a node for each region, you can also add node attributes here based on the region properties
        G.add_node(i)

        # Example: Add edges based on adjacency or other criteria (This is a placeholder)
        # In a real scenario, you might check for spatial adjacency between regions and add edges accordingly

    # Placeholder for adding edges, you might need more sophisticated logic based on your requirements
    return G

# Apply segmentation to all images
segmented_images = {class_name: [segment_image(img) for img in imgs] for class_name, imgs in images.items()}
# Convert all segmented images to graphs
graphs = {class_name: [image_to_graph(img) for img in imgs] for class_name, imgs in segmented_images.items()}