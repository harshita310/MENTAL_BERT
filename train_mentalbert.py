import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 1. Load dataset
dataset = load_dataset(
    "csv",
    data_files={
        "train": "data/raw/mental_heath_unbanlanced.csv",
        "test": "data/raw/mental_health_combined_test.csv"
    }
)

print(dataset)
print(dataset["train"].column_names)
print(dataset["train"][0])


# 2. Clean text and labels
def clean_text(example):
    example["text"] = str(example["text"]).strip()
    example["status"] = str(example["status"]).strip().lower()
    return example

dataset = dataset.map(clean_text)

dataset = dataset.filter(
    lambda x: x["text"] is not None and len(x["text"]) > 0
)


# 3. Label mapping
label_names = sorted(list(set(dataset["train"]["status"])))

label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}

print("Label mapping:")
print(label2id)


def encode_labels(example):
    example["labels"] = label2id[example["status"]]
    return example

dataset = dataset.map(encode_labels)


# 4. Load BERT model for now
# Later replace with MentalBERT if you get access
model_name = "mental/mental-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id
)

# ── FIX: move model to GPU once, here ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on: {device}")
# ───────────────────────────────────────────────────────────────────────────────


# 5. Tokenization
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

encoded_dataset = dataset.map(tokenize, batched=True)

encoded_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


# 6. Metrics
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted")
    }


# 7. Training settings
training_args = TrainingArguments(
    output_dir="./bert_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True
) 

 # ── FIX: tell the Trainer which device to use ──────────────────────────────
no_cuda=not torch.cuda.is_available()   # use GPU when available
    # ───────────────────────────────────────────────────────────────────────────

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics
)


# 9. Train model
trainer.train()


# 10. Evaluate model
results = trainer.evaluate()
print("\nEvaluation Results:")
print(results)

predictions = trainer.predict(encoded_dataset["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_names))


# 11. Save model
save_path = "./models/saved_bert_classifier"

trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("\nModel saved at:", save_path)


# 12. Prediction function
def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
 # ── FIX: move every tensor to the model's device ───────────────────────────
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_id = torch.argmax(probabilities, dim=1).item()

    predicted_label = id2label[predicted_id]
    confidence = probabilities[0][predicted_id].item() * 100

    return predicted_label, confidence


# 13. Sample predictions
sample_texts = [
    "I feel nervous all the time and I cannot stop overthinking.",
    "I feel empty and sad every day.",
    "I want to end everything and disappear forever.",
    "I had a normal day and went out with my friends."
]

print("\nSample Predictions:")

for text in sample_texts:
    label, conf = predict_text(text)
    print("\nText:", text)
    print("Prediction:", label)
    print("Confidence:", round(conf, 2), "%")
