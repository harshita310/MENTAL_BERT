import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


############################################################
# Configuration
############################################################

MODEL_PATH = "models/saved_mentalbert_classifier"

MAX_LENGTH = 128


############################################################
# Device
############################################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MentalBERT Mental Health Classifier")
print("=" * 60)
print(f"Running on: {device}")
print()


############################################################
# Load Model
############################################################

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully.\n")


############################################################
# Label Mapping
############################################################

id2label = {
    int(k): v
    for k, v in model.config.id2label.items()
}


############################################################
# Prediction Function
############################################################

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    predicted_id = torch.argmax(probabilities).item()

    confidence = probabilities[predicted_id].item() * 100

    return predicted_id, confidence, probabilities.cpu().numpy()


############################################################
# Interactive Prediction
############################################################

print("Type 'exit' to quit.\n")

while True:

    text = input("Enter text: ").strip()

    if text.lower() == "exit":
        print("\nExiting...")
        break

    if text == "":
        print("Please enter some text.\n")
        continue

    predicted_id, confidence, probabilities = predict(text)

    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)

    print(f"Input      : {text}")
    print(f"Prediction : {id2label[predicted_id]}")
    print(f"Confidence : {confidence:.2f}%")

    print("\nClass Probabilities")
    print("-" * 60)

    sorted_probs = sorted(
        enumerate(probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    for idx, prob in sorted_probs:
        print(f"{id2label[idx]:<12} : {prob*100:.2f}%")

    print("=" * 60)
    print()