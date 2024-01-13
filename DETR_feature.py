import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

# Ensure you have the necessary libraries installed:
# pip install torch torchvision
# pip install transformers

# Load the model and processor
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

# Load and process the image
image_path = 'datasets/mel_test1.jpg'  # replace with your image path
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Post-process the outputs
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Interpret and print the results
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at {box}")

