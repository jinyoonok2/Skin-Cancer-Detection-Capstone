import os
import re
import yaml

def load_classes_from_yaml(data_folder):
    yaml_path = os.path.join(data_folder, 'data.yaml')
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
        return data.get('names', [])

def truncate_after_jpg(directory, file_extension, classes):
    original_to_new = {}
    for filename in os.listdir(directory):
        base_name = re.sub(r'_jpg.*', '', filename)
        new_name = base_name + file_extension
        counter = 1
        while os.path.exists(os.path.join(directory, new_name)):
            new_name = f"{base_name}-{counter}{file_extension}"
            counter += 1

        if new_name != filename:
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
            original_to_new[filename.rsplit('.', 1)[0]] = new_name.rsplit('.', 1)[0]

    return original_to_new

def capitalize_class_names(directory, original_to_new, file_extension, classes):
    for old_name, new_base_name in original_to_new.items():
        for cls in classes:
            if cls.lower() in new_base_name.lower():
                capitalized_name = re.sub(r'\b' + cls.lower() + r'\b', cls, new_base_name, flags=re.IGNORECASE) + file_extension
                os.rename(os.path.join(directory, new_base_name + file_extension), os.path.join(directory, capitalized_name))
                original_to_new[old_name] = capitalized_name.rsplit('.', 1)[0]
                break

def process_label_files(directory, original_to_new):
    for filename in os.listdir(directory):
        base_name = filename.rsplit('.', 1)[0]
        if base_name in original_to_new:
            new_name = original_to_new[base_name] + '.txt'
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
