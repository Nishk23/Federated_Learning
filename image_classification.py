import os
import random
import shutil


def create_destination_folders(destination_folders):
    for folder_path in destination_folders:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def get_image_list(data_folder, image_extension):
    return [filename for filename in os.listdir(data_folder) if os.path.splitext(filename)[-1] in image_extension]


def split_and_copy_images(data_folder, images_list, train_folder, val_folder, test_folder):
    random.shuffle(images_list)
    train_size = int(len(images_list) * 0.70)
    val_size = int(len(images_list) * 0.15)

    for i, f in enumerate(images_list):
        if i < train_size:
            dest_folder = train_folder
        elif i < train_size + val_size:
            dest_folder = val_folder
        else:
            dest_folder = test_folder
        shutil.copy(os.path.join(data_folder, f), os.path.join(dest_folder, f))


# Set root directory
root = 'C:/Users/nithi/Desktop/FAU/Semester-4/Master_Thesis_Federated_Learning/Dataset'

# Path to image folders
data_folder_atelectasis = os.path.join(root, 'sample/images/Atelectasis')
data_folder_cardiomegaly = os.path.join(root, 'sample/images/Cardiomegaly')

# Path to atelectasis destination folders
train_folder_atelectasis = os.path.join(data_folder_atelectasis, 'train')
val_folder_atelectasis = os.path.join(data_folder_atelectasis, 'eval')
test_folder_atelectasis = os.path.join(data_folder_atelectasis, 'test')

# Path to cardiomegaly destination folders
train_folder_cardiomegaly = os.path.join(data_folder_cardiomegaly, 'train')
val_folder_cardiomegaly = os.path.join(data_folder_cardiomegaly, 'eval')
test_folder_cardiomegaly = os.path.join(data_folder_cardiomegaly, 'test')

# Define a list of image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Create destination folders if they don't exist
create_destination_folders([train_folder_atelectasis, val_folder_atelectasis, test_folder_atelectasis,
                            train_folder_cardiomegaly, val_folder_cardiomegaly, test_folder_cardiomegaly])

# Process Atelectasis images
images_list_atelectasis = get_image_list(data_folder_atelectasis, image_extensions)
split_and_copy_images(data_folder_atelectasis, images_list_atelectasis, train_folder_atelectasis,
                      val_folder_atelectasis, test_folder_atelectasis)

# Process Cardiomegaly images
images_list_cardiomegaly = get_image_list(data_folder_cardiomegaly, image_extensions)
split_and_copy_images(data_folder_cardiomegaly, images_list_cardiomegaly, train_folder_cardiomegaly,
                      val_folder_cardiomegaly, test_folder_cardiomegaly)
