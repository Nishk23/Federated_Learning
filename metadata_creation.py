import os

import pandas as pd

root_path = 'C:/Users/nithi/Desktop/FAU/Semester-4/Master_Thesis_Federated_Learning/dataset'

# Define the path to the dataset
dataset_path = root_path

# Initialize lists to store image IDs and labels
image_ids = []
labels = []

# Iterate through each pathology folder
for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)
    if os.path.isdir(folder_path):
        # Iterate through each image in the folder
        for image_name in os.listdir(folder_path):
            if image_name.endswith('.png'):
                image_ids.append(image_name)
                labels.append(label)

# Create a DataFrame from the lists
metadata_df = pd.DataFrame({
    'Image-ID': image_ids,
    'Label': labels
})

# Define the path to the output CSV file
csv_path = root_path + '/chest_image_metadata.csv'

# Save the DataFrame to a CSV file
metadata_df.to_csv(csv_path, index=False)

print(f"Metadata Excel file created successfully at {csv_path}")
