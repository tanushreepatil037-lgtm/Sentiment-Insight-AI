
from datasets import load_dataset
import os

print("Downloading TweetEval Dataset...")

dataset = load_dataset(
    "cardiffnlp/tweet_eval",
    "sentiment"
)

os.makedirs("dataset", exist_ok=True)

labels = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

for split in ["train", "validation", "test"]:

    df = dataset[split].to_pandas()

    df["sentiment"] = df["label"].map(labels)

    df.to_csv(
        f"dataset/{split}.csv",
        index=False
    )

    print(f"\n{split.upper()}")
    print("Total records:", len(df))
    print(df["sentiment"].value_counts())

print("\nDATASET DOWNLOADED SUCCESSFULLY!")
