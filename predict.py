
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL = "distilbert-base-uncased"
ADAPTER_PATH = "models/sentiment-lora"

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

print("Loading Sentiment Insight AI...")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=3
)

base_model.resize_token_embeddings(len(tokenizer))
base_model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully!")


def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())

    return {
        "text": text,
        "sentiment": LABELS[predicted_id],
        "confidence": round(
            probabilities[predicted_id].item() * 100, 2
        ),
        "probabilities": {
            LABELS[i]: round(probabilities[i].item() * 100, 2)
            for i in range(3)
        }
    }


if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("SENTIMENT INSIGHT AI - PREDICTION")
    print("=" * 50)

    while True:

        user_text = input(
            "\nEnter text (or type 'exit' to stop): "
        )

        if user_text.lower().strip() == "exit":
            print("Thank you for using Sentiment Insight AI!")
            break

        if not user_text.strip():
            print("Please enter some text.")
            continue

        result = predict_sentiment(user_text)

        print("\nPREDICTION RESULT")
        print("-" * 40)
        print("Text:", result["text"])
        print("Sentiment:", result["sentiment"])
        print("Confidence:", str(result["confidence"]) + "%")

        print("\nClass probabilities:")

        for label, probability in result["probabilities"].items():
            print(f"{label}: {probability}%")
