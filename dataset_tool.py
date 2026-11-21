
import os
import pandas as pd


# Dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train.csv"
)


def analyze_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    if "sentiment" not in df.columns:
        raise ValueError("Sentiment column is missing.")

    # Calculate sentiment counts
    sentiment_counts = df["sentiment"].value_counts()

    total_records = len(df)

    # Calculate percentages
    sentiment_percentages = (
        sentiment_counts / total_records * 100
    ).round(2)

    # Find dominant sentiment
    dominant_sentiment = sentiment_counts.idxmax()

    # Dataset analysis
    result = {
        "total_records": total_records,
        "sentiment_counts": sentiment_counts.to_dict(),
        "sentiment_percentages": sentiment_percentages.to_dict(),
        "dominant_sentiment": dominant_sentiment
    }

    return result


# Direct testing
if __name__ == "__main__":

    result = analyze_dataset()

    print("\nDATASET ANALYSIS")
    print("=" * 35)

    print("Total Records:", result["total_records"])

    print("\nSentiment Distribution:")

    for sentiment, count in result["sentiment_counts"].items():
        percentage = result["sentiment_percentages"][sentiment]

        print(
            f"{sentiment}: {count} ({percentage}%)"
        )

    print(
        "\nDominant Sentiment:",
        result["dominant_sentiment"]
    )
