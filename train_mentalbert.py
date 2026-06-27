import os
import json
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


############################################################
# Configuration
############################################################

MODEL_NAME = "mental/mental-bert-base-uncased"

TRAIN_FILE = "data/raw/mental_heath_unbanlanced.csv"
TEST_FILE = "data/raw/mental_health_combined_test.csv"

MODEL_SAVE_PATH = "models/saved_mentalbert_classifier"

RESULTS_DIR = "results"

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

LEARNING_RATE = 2e-5

NUM_EPOCHS = 2

WEIGHT_DECAY = 0.01

OUTPUT_DIR = "bert_results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


############################################################
# Device
############################################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", device)

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))

print("=" * 60)


############################################################
# Load Dataset
############################################################

print("\nLoading Dataset...\n")

dataset = load_dataset(
    "csv",
    data_files={
        "train": TRAIN_FILE,
        "test": TEST_FILE
    }
)

print(dataset)

print("\nTraining Columns:")
print(dataset["train"].column_names)

print("\nFirst Sample:")
print(dataset["train"][0])
############################################################
# Data Cleaning
############################################################

print("\nCleaning Dataset...\n")


def clean_text(example):
    example["text"] = str(example["text"]).strip()
    example["status"] = str(example["status"]).strip().lower()
    return example


dataset = dataset.map(clean_text)

dataset = dataset.filter(
    lambda x: x["text"] is not None and len(x["text"]) > 0
)

print("Dataset cleaned successfully.")


############################################################
# Label Encoding
############################################################

print("\nCreating Label Mapping...\n")

label_names = sorted(list(set(dataset["train"]["status"])))

label2id = {
    label: idx
    for idx, label in enumerate(label_names)
}

id2label = {
    idx: label
    for label, idx in label2id.items()
}

print("Label Mapping")

for label, idx in label2id.items():
    print(f"{label:15} --> {idx}")


def encode_labels(example):
    example["labels"] = label2id[example["status"]]
    return example


dataset = dataset.map(encode_labels)

print("\nLabels encoded successfully.")


############################################################
# Load MentalBERT
############################################################

print("\nLoading MentalBERT...\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    label2id=label2id,
    id2label=id2label
)

model.to(device)

print("MentalBERT loaded successfully.")


############################################################
# Tokenization
############################################################

print("\nTokenizing Dataset...\n")


def tokenize(batch):

    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


encoded_dataset = dataset.map(
    tokenize,
    batched=True
)


encoded_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

print("Tokenization Complete.")

print("\nTraining Samples :", len(encoded_dataset["train"]))
print("Testing Samples  :", len(encoded_dataset["test"]))


############################################################
# Metrics
############################################################

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }
############################################################
# Training Arguments
############################################################

print("\nInitializing Trainer...\n")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    logging_steps=50,

    load_best_model_at_end=True,

    report_to="none"
)


############################################################
# Trainer
############################################################

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics
)


############################################################
# Train Model
############################################################

print("\nStarting Fine-Tuning...\n")

trainer.train()

print("\nTraining Completed Successfully.")


############################################################
# Evaluate Model
############################################################

print("\nEvaluating Model...\n")

results = trainer.evaluate()

print("\nEvaluation Results")
print(results)


############################################################
# Predictions
############################################################

predictions = trainer.predict(encoded_dataset["test"])

y_pred = np.argmax(predictions.predictions, axis=1)

y_true = predictions.label_ids


############################################################
# Classification Report
############################################################

report = classification_report(
    y_true,
    y_pred,
    target_names=label_names,
    digits=4
)

print("\nClassification Report\n")
print(report)

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:
    f.write(report)


############################################################
# Save Metrics
############################################################

metrics = {

    "accuracy": float(results["eval_accuracy"]),

    "macro_f1": float(results["eval_macro_f1"]),

    "weighted_f1": float(results["eval_weighted_f1"]),

    "loss": float(results["eval_loss"])

}

with open(
    os.path.join(
        RESULTS_DIR,
        "metrics.json"
    ),
    "w"
) as f:
    json.dump(metrics, f, indent=4)


############################################################
# Confusion Matrix
############################################################

cm = confusion_matrix(
    y_true,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=label_names,
    columns=label_names
)

cm_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.csv"
    )
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_names
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(
    cmap="Blues",
    ax=ax,
    values_format="d"
)

plt.title("MentalBERT Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


############################################################
# Save Label Mapping
############################################################

with open(
    os.path.join(
        RESULTS_DIR,
        "label_mapping.json"
    ),
    "w"
) as f:
    json.dump(
        label2id,
        f,
        indent=4
    )


############################################################
# Training Summary
############################################################

summary = f"""
===============================
MentalBERT Fine-Tuning Summary
===============================

Model
-----
{MODEL_NAME}

Training Samples
----------------
{len(encoded_dataset['train'])}

Testing Samples
---------------
{len(encoded_dataset['test'])}

Epochs
------
{NUM_EPOCHS}

Learning Rate
-------------
{LEARNING_RATE}

Batch Size
----------
{TRAIN_BATCH_SIZE}

Maximum Sequence Length
-----------------------
{MAX_LENGTH}

Evaluation Metrics
------------------

Accuracy     : {results['eval_accuracy']:.4f}

Macro F1     : {results['eval_macro_f1']:.4f}

Weighted F1  : {results['eval_weighted_f1']:.4f}

Loss         : {results['eval_loss']:.4f}

"""

with open(
    os.path.join(
        RESULTS_DIR,
        "training_summary.txt"
    ),
    "w"
) as f:
    f.write(summary)


############################################################
# Save Model
############################################################

trainer.save_model(MODEL_SAVE_PATH)

tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("\nModel Saved Successfully.")

print(MODEL_SAVE_PATH)
############################################################
# Prediction Function
############################################################

def predict_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_id = torch.argmax(
            probabilities,
            dim=1
        ).item()

    predicted_label = id2label[predicted_id]

    confidence = probabilities[0][predicted_id].item()

    return predicted_label, confidence, probabilities.cpu().numpy()


############################################################
# Sample Predictions
############################################################

sample_texts = [

    "I feel nervous all the time and I cannot stop overthinking.",

    "I feel empty and sad every day.",

    "I want to end everything and disappear forever.",

    "I had a normal day and went out with my friends."
]

prediction_file = os.path.join(
    RESULTS_DIR,
    "sample_predictions.txt"
)

with open(prediction_file, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("MentalBERT Sample Predictions\n")
    f.write("=" * 70 + "\n\n")

    for i, text in enumerate(sample_texts, start=1):

        label, confidence, _ = predict_text(text)

        result = (
            f"Sample {i}\n"
            f"Input      : {text}\n"
            f"Prediction : {label}\n"
            f"Confidence : {confidence*100:.2f}%\n"
            + "-" * 70 + "\n"
        )

        print(result)

        f.write(result)

print("\nSample predictions saved successfully.")

print(f"\nResults saved to: {RESULTS_DIR}")

print("\nTraining pipeline completed successfully.")