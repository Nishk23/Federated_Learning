import os
import random
import shutil


def setup_and_copy_images(data_folder, train_folder, val_folder, test_folder, image_extension):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    images_list = [f for f in os.listdir(data_folder) if os.path.splitext(f)[-1] in image_extension]
    random.shuffle(images_list)

    train_size = int(len(images_list) * 0.70)
    val_size = int(len(images_list) * 0.15)

    for i, f in enumerate(images_list):
        dest_folder = (train_folder if i < train_size else val_folder if i < train_size + val_size else test_folder)
        shutil.copy(os.path.join(data_folder, f), os.path.join(dest_folder, f))


root = 'C:/Users/nithi/Desktop/FAU/Semester-4/Master_Thesis_Federated_Learning/Dataset'
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Atelectasis
data_folder_atelectasis = os.path.join(root, 'sample/images/Atelectasis')
train_folder_atelectasis = os.path.join(data_folder_atelectasis, 'train')
val_folder_atelectasis = os.path.join(data_folder_atelectasis, 'eval')
test_folder_atelectasis = os.path.join(data_folder_atelectasis, 'test')
setup_and_copy_images(data_folder_atelectasis, train_folder_atelectasis, val_folder_atelectasis,
                      test_folder_atelectasis, image_extensions)

# Cardiomegaly
data_folder_cardiomegaly = os.path.join(root, 'sample/images/Cardiomegaly')
train_folder_cardiomegaly = os.path.join(data_folder_cardiomegaly, 'train')
val_folder_cardiomegaly = os.path.join(data_folder_cardiomegaly, 'eval')
test_folder_cardiomegaly = os.path.join(data_folder_cardiomegaly, 'test')
setup_and_copy_images(data_folder_cardiomegaly, train_folder_cardiomegaly, val_folder_cardiomegaly,
                      test_folder_cardiomegaly, image_extensions)
