import os
import random


def reduce_images(folder_path, max_images=1000):
    # Get a list of all files in the folder
    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Filter only image files (assuming images are in common formats)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    image_files = [
        f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions
    ]

    # If the number of images is less than or equal to the max_images, do nothing
    if len(image_files) <= max_images:
        print(f"{folder_path} already has {len(image_files)} or fewer images.")
        return

    # Randomly shuffle the list of image files
    random.shuffle(image_files)

    # Select the images to keep
    # images_to_keep = image_files[:max_images]

    # Select the images to delete
    images_to_delete = image_files[max_images:]

    # Delete the selected images
    for img in images_to_delete:
        os.remove(img)
        print(f"Deleted: {img}")

    print(f"Reduced {folder_path} to {max_images} images.")


# Example usage
folders = [
    "Dataset/sample/Atelectasis",
    "Dataset/sample/Cardiomegaly",
    "Dataset/sample/Consolidation",
    "Dataset/sample/Edema",
    "Dataset/sample/Effusion",
    "Dataset/sample/Emphysema",
    "Dataset/sample/Fibrosis",
    "Dataset/sample/Hernia",
    "Dataset/sample/Infiltration",
    "Dataset/sample/Mass",
    "Dataset/sample/Nodule",
    "Dataset/sample/Pleural_Thickening",
    "Dataset/sample/Pneumonia",
    "Dataset/sample/Pneumothorax",
    "Dataset/sample/No Finding",
    # Add more folders as needed
]

for folder in folders:
    reduce_images(folder, max_images=900)
