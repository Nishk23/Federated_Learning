import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoImageProcessor,
    ResNetForImageClassification,
    TrainingArguments,
    Trainer,
)

# Paths
data_path = "/home/hpc/rzku/mlvl138h/Federated_Learning/Dataset/sample"
csv_path = os.path.join(data_path, "chest_image_metadata.csv")

# Load metadata
metadata = pd.read_csv(csv_path)


# Custom Dataset class
class PathologyDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.img_dir, self.metadata.iloc[idx, 1], self.metadata.iloc[idx, 0]
        )
        image = Image.open(img_name).convert("RGB")
        label = classes.index(self.metadata.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return {"pixel_values": inputs["pixel_values"], "labels": torch.tensor(label)}


# Define class names
classes = [
    "Hernia",
    "Pneumonia",
    "Fibrosis",
    "Nodule",
    "Mass",
    "Consolidation",
    "Effusion",
    "Edema",
    "Atelectasis",
    "No Finding",
    "Cardiomegaly",
    "Pneumothorax",
    "Pleural_Thickening",
    "Infiltration",
    "Emphysema",
]

# Initialize processor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# Create dataset
dataset = PathologyDataset(csv_file=csv_path, img_dir=data_path, processor=processor)

# Split dataset into train, validation, and test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Load and modify the model
model = ResNetForImageClassification.from_pretrained(
    "microsoft/resnet-50", num_labels=len(classes), ignore_mismatched_sizes=True
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output/checkpoints",  # Ensure outputs are saved in the output/checkpoints directory
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)


# Define compute metrics function
def compute_metrics(p):
    pred_labels = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, pred_labels, average="weighted"
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Add progress bar for training
trainer.train()

# Evaluate the model on the test set
print("Evaluating model...")
test_results = trainer.evaluate(test_dataset)
print(test_results)

# Predict and evaluate on the test set
print("Predicting on test set...")
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)
true_labels = [
    example["labels"].item()
    for example in tqdm(test_dataset, desc="Loading test dataset")
]

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
ConfusionMatrixDisplay(conf_matrix, display_labels=classes).plot(
    xticks_rotation="vertical"
)
plt.show()


# Function to predict and display an image
def predict_and_display(img_path, true_label):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label_idx = logits.argmax(-1).item()
    predicted_label = classes[predicted_label_idx]

    plt.imshow(image)
    plt.title(f"True Label: {true_label}\nPredicted Label: {predicted_label}")
    plt.axis("off")
    plt.show()


# Example usage
example_img_path = os.path.join(data_path, metadata.iloc[0, 1], metadata.iloc[0, 0])
true_label = metadata.iloc[0, 1]
predict_and_display(example_img_path, true_label)
