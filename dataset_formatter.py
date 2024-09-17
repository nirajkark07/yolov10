import os
import shutil
import random
from PIL import Image
import json
import shutil

# Defintion
dataset_dir = r'E:\NIRAJ\Datasets\2024-09-16 GearBox\Niraj_2'  # Change this to the directory where your dataset is stored
output_dir = r'E:\NIRAJ\GIT\yolov10\dataset'  # Change this to where you want the train/valid/test folders to be created
train_split = 0.7
valid_split = 0.2
test_split = 0.1

# Ensure the split ratios sum to 1
assert train_split + valid_split + test_split != 1.0, "Split ratios should sum to 1."

# Create output directories if they don't exist
for folder in ['train', 'valid', 'test']:
    for subfolder in ['images', 'labels']:
        os.makedirs(os.path.join(output_dir, folder, subfolder), exist_ok=True)

# Get a list of all .png files (images) and corresponding .json files (labels)
image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
json_files = [f.replace('.png', '.json') for f in image_files]

# Ensure there's a corresponding json file for each image
assert all(os.path.exists(os.path.join(dataset_dir, json_file)) for json_file in json_files), "Some JSON files are missing."

# Shuffle the dataset to ensure randomness in the split
combined_files = list(zip(image_files, json_files))
random.shuffle(combined_files)

# Calculate the number of samples for each split
total_images = len(combined_files)
train_count = int(train_split * total_images)
valid_count = int(valid_split * total_images)
test_count = total_images - train_count - valid_count

# Split the data
train_files = combined_files[:train_count]
valid_files = combined_files[train_count:train_count + valid_count]
test_files = combined_files[train_count + valid_count:]

# Helper function to copy files to destination folder
def copy_files(file_pairs, destination_folder):
    for image_file, json_file in file_pairs:
        # Copy image and corresponding json file
        shutil.copy(os.path.join(dataset_dir, image_file), os.path.join(destination_folder, 'images', image_file))
        shutil.copy(os.path.join(dataset_dir, json_file), os.path.join(destination_folder, 'images', json_file))

# Copy files to respective folders
copy_files(train_files, os.path.join(output_dir, 'train'))
copy_files(valid_files, os.path.join(output_dir, 'valid'))
copy_files(test_files, os.path.join(output_dir, 'test'))


# Iterate over train, valid, and test folders
for split in ['train', 'valid', 'test']:
    images_dir = os.path.join(output_dir, split, 'images')
    labels_dir = os.path.join(output_dir, split, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # Get all .json files in the images folder
    json_files = [f for f in os.listdir(images_dir) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(images_dir, json_file)
        image_file = json_file.replace('.json', '.png')

        # Read the corresponding image to get its dimensions
        image_path = os.path.join(images_dir, image_file)
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading image {image_file}: {e}")
            continue

        # Read the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Create a corresponding .txt file in the labels folder
        label_file = os.path.join(labels_dir, json_file.replace('.json', '.txt'))
        with open(label_file, 'w') as label_f:
            for obj in data['objects']:
                class_id = obj['instance_id']  # Use instance_id as class_id

                # Get bounding box coordinates
                top_left = obj['bounding_box']['top_left']
                bottom_right = obj['bounding_box']['bottom_right']

                # Convert bounding box to YOLO format (center_x, center_y, width, height)
                x_min = top_left[0]
                y_min = top_left[1]
                x_max = bottom_right[0]
                y_max = bottom_right[1]

                # Calculate normalized (center_x, center_y, width, height)
                center_x = ((x_min + x_max) / 2) / img_width
                center_y = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # Write the class_id and normalized bounding box to the label file
                label_f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

        # Move Json file to old directory.
        os.remove(json_path)

print("Conversion complete. Labels saved in YOLO format.")