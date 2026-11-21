
import os
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_MODEL = "distilbert-base-uncased"

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "sentiment-lora"
)

LABELS = ["Negative", "Neutral", "Positive"]

DEVICE = torch.device("cpu")

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

tokenizer = None
model = None


def load_sentiment_model():
    global tokenizer, model

    if model is not None:
        return

    print("Loading trained sentiment model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model adapter not found: {MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3
    )

    model = PeftModel.from_pretrained(
        base_model,
        MODEL_PATH
    )

    model.to(DEVICE)
    model.eval()

    print("Sentiment model loaded successfully!")


# --------------------------------------------------
# SENTIMENT PREDICTION
# --------------------------------------------------

def predict_sentiment(text):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter valid text.")

    load_sentiment_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    predicted_index = int(torch.argmax(probabilities))

    result = {
        "sentiment": LABELS[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index]) * 100, 2
        ),
        "probabilities": {
            LABELS[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(3)
        }
    }

    return result


# --------------------------------------------------
# DIRECT TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_texts = [
        "I am very happy today!",
        "This product is terrible.",
        "The meeting is scheduled for tomorrow."
    ]

    for text in test_texts:

        print("\nText:", text)

        try:
            result = predict_sentiment(text)

            print("Sentiment:", result["sentiment"])
            print("Confidence:", result["confidence"], "%")
            print("Probabilities:", result["probabilities"])

        except Exception as error:
            print("Error:", error)
