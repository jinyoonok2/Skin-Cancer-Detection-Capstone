import os
import shutil
from sklearn.model_selection import train_test_split


def distribute_files():
    nv_dir = 'NV'  # The directory containing NV images and labels
    augmented_dir = 'augmented'  # The directory where train/valid folders are located

    # Collecting all unique image-label pairs
    unique_pairs = []
    for filename in os.listdir(nv_dir):
        if filename.endswith('.jpg') or filename.endswith('.txt'):
            # Extract the base name without extension and class suffix
            base_name = '_'.join(filename.split('_')[:-1])
            if base_name not in unique_pairs:
                unique_pairs.append(base_name)

    # Splitting the pairs into train and valid sets
    train_pairs, valid_pairs = train_test_split(unique_pairs, test_size=0.2, random_state=42)

    # Function to move files to the respective directories
    def move_files(file_pairs, dest_subdir):
        for base_name in file_pairs:
            img_src = os.path.join(nv_dir, f"{base_name}_NV.jpg")
            lbl_src = os.path.join(nv_dir, f"{base_name}_NV.txt")

            img_dest = os.path.join(augmented_dir, dest_subdir, 'images', f"{base_name}_NV.jpg")
            lbl_dest = os.path.join(augmented_dir, dest_subdir, 'labels', f"{base_name}_NV.txt")

            if os.path.exists(img_src) and os.path.exists(lbl_src):
                shutil.move(img_src, img_dest)
                shutil.move(lbl_src, lbl_dest)

    # Creating necessary directories if they don't exist
    os.makedirs(os.path.join(augmented_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(augmented_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(augmented_dir, 'valid', 'images'), exist_ok=True)
    os.makedirs(os.path.join(augmented_dir, 'valid', 'labels'), exist_ok=True)

    # Distributing files to train and valid directories
    move_files(train_pairs, 'train')
    move_files(valid_pairs, 'valid')

    print(f"Finished distributing {len(train_pairs)} training and {len(valid_pairs)} validation pairs.")


# Run the function
distribute_files()
