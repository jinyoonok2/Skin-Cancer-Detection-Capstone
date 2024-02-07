import yaml
import os
from dataset_path import YAML_PATH, DEFAULT_PATH

def are_similar(coord1, coord2, tolerance=0.03):
    """Check if two coordinates are similar within a given tolerance."""
    return abs(coord1 - coord2) <= tolerance * max(coord1, coord2)

def is_duplicate(line1, line2):
    """Check if two lines are considered duplicates based on the first 4 coordinates."""
    coords1 = [float(x) for x in line1.split()[1:21]]
    coords2 = [float(x) for x in line2.split()[1:21]]

    return all(are_similar(coord1, coord2) for coord1, coord2 in zip(coords1, coords2))

def remove_duplicates_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Remove duplicates based on similarity of the first 4 coordinates
    unique_lines = []
    for line in lines:
        duplicate_found = False
        for other_line in unique_lines:
            if is_duplicate(line, other_line):
                print(f"Duplicate found in {filepath}:")
                print(f"  Original line: {other_line.strip()} (Coords: {' '.join(other_line.split()[1:21])})")
                print(f"  Duplicate line: {line.strip()} (Coords: {' '.join(line.split()[1:21])})")
                duplicate_found = True
                break

        if not duplicate_found:
            unique_lines.append(line)

    # Write the unique lines back to the file
    with open(filepath, 'w') as file:
        file.writelines(unique_lines)

# Load class names from the YAML file
with open(YAML_PATH, 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']

# Process each label file in each class folder
for class_name in class_names:
    labels_path = os.path.join(DEFAULT_PATH, class_name, 'labels')
    for filename in os.listdir(labels_path):
        file_path = os.path.join(labels_path, filename)
        remove_duplicates_from_file(file_path)

print("Duplicate removal process completed.")
