import shutil
from pathlib import Path

# Define the path to your dataset
dataset_path = Path("C:/Users/nithi/Desktop/FAU/Semester-4/Master_Thesis_Federated_Learning/dataset")
client_data_paths = [Path(f"C:/Users/nithi/Desktop/FAU/Semester-4/Master_Thesis_Federated_Learning/workspace/"
                          f"Fl_NV_Flare/client_{i}_data") for i in range(1, 4)]

# Ensure the client directories exist
for client_path in client_data_paths:
    if client_path.exists():
        shutil.rmtree(client_path)
    client_path.mkdir(parents=True, exist_ok=True)

# Iterate through each pathology folder and distribute images
for pathology_folder in dataset_path.iterdir():
    if pathology_folder.is_dir():
        # List all image files in the pathology folder
        image_files = list(pathology_folder.glob('*.png'))

        # Split data equally among 3 clients
        client_data_splits = [image_files[i::3] for i in range(3)]

        # Ensure each client has a subfolder for the current pathology
        for client_index, client_files in enumerate(client_data_splits):
            client_path = client_data_paths[client_index] / pathology_folder.name
            client_path.mkdir(parents=True, exist_ok=True)

            # Copy files to the respective client pathology directories
            for file in client_files:
                shutil.copy(file, client_path)

print("Data has been distributed equally among the 3 clients.")
