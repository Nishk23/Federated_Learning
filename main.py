import os
import random
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# Function to display images along with their file names and ages
def display_images_with_metadata(image_paths, image_data):
    plt.figure(figsize=(20, 10))
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        file_name = os.path.basename(image_path)
        age = image_data.loc[image_data['Image-ID'] == file_name, 'Age'].values[0]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f"{file_name}\nAge: {age}")  # Display the file name and age
        plt.axis('off')
    plt.show()


# Path to your image folder
image_folder = 'C:/Users/nithi/Desktop/FAU/Semester-4/Master_Thesis_Federated_Learning/Dataset/sample/images'

# Path to your CSV file containing metadata
csv_file_path = 'C:/Users/nithi/Desktop/FAU/Semester-4/Master_Thesis_Federated_Learning/Dataset/sample_labels.csv'

# Read the metadata from the CSV file
metadata = pd.read_csv(csv_file_path)

# Extract only numeric part from the 'age' column
metadata['Age'] = metadata['Age'].str.extract('(\d+)').astype(int)

# Get list of all image files in the folder
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if
               file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

# Select 10 random images
random_images = random.sample(image_files, 10)

# Display the images with file names and ages
display_images_with_metadata(random_images, metadata)
