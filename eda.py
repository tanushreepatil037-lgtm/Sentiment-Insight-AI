
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Professional EDA for Sentiment Analysis LLM

print("=" * 60)
print("SENTIMENT INSIGHT AI - DATA ANALYSIS")
print("=" * 60)

os.makedirs("reports/figures", exist_ok=True)

# Load datasets
train = pd.read_csv("dataset/train.csv")
validation = pd.read_csv("dataset/validation.csv")
test = pd.read_csv("dataset/test.csv")

# Dataset overview
print("\nDATASET SHAPE")
print("Training:", train.shape)
print("Validation:", validation.shape)
print("Testing:", test.shape)

print("\nMISSING VALUES")
print(train.isnull().sum())

print("\nDUPLICATE RECORDS")
print("Training:", train.duplicated(subset=["text"]).sum())

# Sentiment distribution
distribution = train["sentiment"].value_counts()

print("\nSENTIMENT DISTRIBUTION")
print(distribution)

# Calculate percentages
percentages = train["sentiment"].value_counts(normalize=True) * 100
print("\nCLASS PERCENTAGES")
print(percentages.round(2))

# Visualization 1: Class distribution
plt.figure(figsize=(9, 5))
sns.countplot(
    data=train,
    x="sentiment",
    hue="sentiment",
    order=["Negative", "Neutral", "Positive"],
    palette="viridis",
    legend=False
)
plt.title("Training Dataset - Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("reports/figures/sentiment_distribution.png")
plt.close()

# Visualization 2: Pie chart
fig = px.pie(
    values=distribution.values,
    names=distribution.index,
    title="Sentiment Class Distribution",
    hole=0.4
)
fig.write_html("reports/figures/sentiment_pie.html")

# Text length analysis
train["text_length"] = train["text"].astype(str).apply(len)

print("\nTEXT LENGTH STATISTICS")
print(train["text_length"].describe())

# Visualization 3: Text length distribution
plt.figure(figsize=(9, 5))
sns.histplot(
    train["text_length"],
    bins=40,
    kde=True
)
plt.title("Text Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reports/figures/text_length_distribution.png")
plt.close()

# Sentiment-wise text length
print("\nAVERAGE TEXT LENGTH BY SENTIMENT")
print(train.groupby("sentiment")["text_length"].mean())

# Sample records
print("\nSAMPLE RECORDS")
print(train[["text", "sentiment"]].head(10).to_string(index=False))

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts saved in reports/figures/")
