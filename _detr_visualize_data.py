from _detr_custom_dataloader import TRAIN_DATASET
import numpy as np
import cv2
import random
import os

def draw_segmentation_masks_and_labels(image, annotations):
    for annotation in annotations:
        # Extract segmentation data
        segmentation = annotation.get('segmentation', None)
        if segmentation:
            # Segmentation format: [[x1, y1, x2, y2, ..., xn, yn]]
            # Draw each segment
            for segment in segmentation:
                poly = np.array(segment).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(image, [poly], True, (0, 255, 0), 2)

        # Optionally, you can also add category labels as done previously

    return image

if __name__ == "__main__":
    # Select a random image and its annotations
    image_ids = TRAIN_DATASET.coco.getImgIds()
    selected_image_id = random.choice(image_ids)
    image_info = TRAIN_DATASET.coco.loadImgs(selected_image_id)[0]
    annotations = TRAIN_DATASET.coco.loadAnns(TRAIN_DATASET.coco.getAnnIds(imgIds=[selected_image_id]))

    # Load image
    image_path = os.path.join(TRAIN_DATASET.root, image_info['file_name'])
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Draw segmentation masks and labels on the image
    annotated_image = draw_segmentation_masks_and_labels(image, annotations)

    # Display the image
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()