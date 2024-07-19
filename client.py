# client.py
import os
import sys

# Set the TRANSFORMERS_CACHE environment variable
os.environ['HF_HOME'] = '/home/cip/nf2024/oq06uxaq/.cache/huggingface/hub'

import flwr as fl
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import random_split
from transformers import AutoFeatureExtractor, ResNetForImageClassification, TrainingArguments, Trainer


# Custom dataset class from image_classification.py
class PathologyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, processor, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx, 1], self.metadata.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = classes.index(self.metadata.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return {'pixel_values': inputs['pixel_values'], 'labels': torch.tensor(label)}


# Define class names
classes = ['Hernia', 'Pneumonia', 'Fibrosis', 'Nodule', 'Mass', 'Consolidation', 'Effusion', 'Edema', 'Atelectasis',
           'No Finding', 'Cardiomegaly', 'Pneumothorax', 'Pleural_Thickening', 'Infiltration', 'Emphysema']


# Compute metrics function from image_classification.py
def compute_metrics(p):
    pred_labels = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, pred_labels, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# Function to load data from a specific folder
def load_data(data_dir, csv_file):
    processor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50",
                                                     force_download=False)
    dataset = PathologyDataset(csv_file=csv_file, img_dir=data_dir, processor=processor)

    # Split dataset into train, validation, and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


# Define Flower client
class ImageClassificationClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, val_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.trainer = self.create_trainer()

    def create_trainer(self):
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=0.001,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,  # Set to 1 for federated learning rounds
            weight_decay=0.01,
        )
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.trainer.train()
        return self.get_parameters(config), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.trainer.evaluate()
        return metrics["eval_loss"], len(self.val_dataset), {"accuracy": metrics["eval_accuracy"],
                                                             "loss": metrics["eval_loss"]}

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(param) for param in parameters]))
        self.model.load_state_dict(state_dict, strict=True)


def main(data_dir):
    # Paths
    csv_file = os.path.join(data_dir, 'chest_image_metadata_client.csv')  # Update as per client

    # Load model
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50",
                                                         num_labels=len(classes),
                                                         ignore_mismatched_sizes=True)

    # Load data
    train_dataset, val_dataset, test_dataset = load_data(data_dir, csv_file)

    # Start Flower client
    client = ImageClassificationClient(model, train_dataset, val_dataset)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <data_dir>")
        sys.exit(1)
    data_dir = sys.argv[1]  # Pass the data directory as a command line argument
    main(data_dir)
