import yaml
import matplotlib.pyplot as plt
from dataset_path import *

# Load class names from the YAML file
with open(YAML_PATH, 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']

# Count the number of images in each class
image_counts = {}
total_images = 0
for class_name in class_names:
    images_path = os.path.join(DEFAULT_PATH, class_name, 'images')
    count = len([name for name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, name))])
    image_counts[class_name] = count
    total_images += count

# Calculate the percentage of images for each class
percentages = {class_name: (count / total_images * 100) for class_name, count in image_counts.items()}

# Print the number and percentage of images for each class
for class_name, count in image_counts.items():
    print(f"Class: {class_name}, Count: {count}, Percentage: {percentages[class_name]:.2f}%")

# Create a pie chart
labels = percentages.keys()
sizes = percentages.values()

plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of Images per Class')

# Save the diagram as a PNG file
save_dir = os.path.join(DATASET_DIR, 'class_distribution.png')
plt.savefig(save_dir)

# Display the plot
plt.show()
