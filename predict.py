from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load saved model
model_path = "./models/saved_bert_classifier"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)



# ── Device setup — FIX: move model to GPU before inference ─────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running inference on: {device}")
# ───────────────────────────────────────────────────────────────────────────────


# Label mapping
id2label = {
    0: "anxiety",
    1: "depression",
    2: "normal",
    3: "suicidal"
}

# User input
text = input("Enter text: ")

# Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

# Move input to same device as model
inputs = {k: v.to(device) for k, v in inputs.items()}

# Prediction
model.eval()

with torch.no_grad():
    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    predicted_id = torch.argmax(probs, dim=1).item()

    confidence = probs[0][predicted_id].item() * 100

# Output
print("\nPrediction Results")
print("------------------")
print("Predicted Class ID :", predicted_id)
print("Predicted Label    :", id2label[predicted_id])
print("Confidence         :", round(confidence, 2), "%")


# Show all class probabilities
print("\nAll Class Probabilities:")
for idx, prob in enumerate(probs[0]):
    print(f"  {id2label[idx]:<12} : {round(prob.item() * 100, 2)} %")