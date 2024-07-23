import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import random_split
from transformers import AutoFeatureExtractor, ResNetForImageClassification, TrainingArguments, Trainer


# Custom dataset class (same as in client.py)
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


# Define class names (same as in client.py)
classes = ['Hernia', 'Pneumonia', 'Fibrosis', 'Nodule', 'Mass', 'Consolidation', 'Effusion', 'Edema', 'Atelectasis',
           'No Finding', 'Cardiomegaly', 'Pneumothorax', 'Pleural_Thickening', 'Infiltration', 'Emphysema']


# Compute metrics function (same as in client.py)
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


# Function to load data (same as in client.py)
def load_data(data_dir, csv_file):
    processor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50", force_download=False)
    dataset = PathologyDataset(csv_file=csv_file, img_dir=data_dir, processor=processor)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


def train_isolated_model(data_dir, client_id):
    csv_file = os.path.join(data_dir, 'chest_image_metadata_client.csv')

    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels=len(classes),
                                                         ignore_mismatched_sizes=True)

    train_dataset, val_dataset, test_dataset = load_data(data_dir, csv_file)

    training_args = TrainingArguments(
        output_dir=f"./results/client_{client_id}_isolated",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=0.001,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,  # Set to a reasonable number of epochs
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train and evaluate the model
    trainer.train()
    metrics = trainer.evaluate()

    # Log the metrics
    metrics_log = {
        "client_id": client_id,
        "accuracy": metrics["eval_accuracy"],
        "precision": metrics["eval_precision"],
        "recall": metrics["eval_recall"],
        "f1": metrics["eval_f1"]
    }
    with open(f"isolated_metrics_client_{client_id}.json", "w") as f:
        json.dump(metrics_log, f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_isolated.py <data_dir> <client_id>")
        sys.exit(1)
    data_dir = sys.argv[1]
    client_id = sys.argv[2]
    train_isolated_model(data_dir, client_id)
