
import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from peft import LoraConfig, get_peft_model, TaskType

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/sentiment-lora"
REPORT_DIR = "reports"

LABEL_MAP = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("SENTIMENT INSIGHT AI - IMPROVED TRAINING")
print("=" * 60)

print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")

print("\nOriginal training distribution:")
print(train_df["sentiment"].value_counts())

# ==========================================
# STEP 2: BALANCE TRAINING DATA
# ==========================================

samples_per_class = 1666

balanced_parts = []

for sentiment in ["Negative", "Neutral", "Positive"]:

    class_data = train_df[
        train_df["sentiment"] == sentiment
    ]

    selected = class_data.sample(
        n=min(samples_per_class, len(class_data)),
        random_state=42
    )

    balanced_parts.append(selected)

train_df = pd.concat(
    balanced_parts,
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced training distribution:")
print(train_df["sentiment"].value_counts())

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(val_df))

# ==========================================
# STEP 3: PREPARE LABELS
# ==========================================

train_df["label"] = train_df["sentiment"].map(LABEL_MAP)
val_df["label"] = val_df["sentiment"].map(LABEL_MAP)

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]].reset_index(drop=True)
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]].reset_index(drop=True)
)

# ==========================================
# STEP 4: LOAD TOKENIZER
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]"
    })

tokenizer.padding_side = "right"

# ==========================================
# STEP 5: LOAD PRETRAINED MODEL
# ==========================================

print("\nLoading pretrained DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID_TO_LABEL,
    label2id=LABEL_MAP
)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# ==========================================
# STEP 6: TOKENIZATION
# ==========================================

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )

print("\nTokenizing data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])

# ==========================================
# STEP 7: LoRA CONFIGURATION
# ==========================================

print("\nApplying LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
    modules_to_save=["pre_classifier", "classifier"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# STEP 8: METRICS
# ==========================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ==========================================
# STEP 9: TRAINING SETTINGS
# ==========================================

training_args = TrainingArguments(
    output_dir="reports/improved_checkpoints",

    num_train_epochs=1,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    learning_rate=2e-4,
    weight_decay=0.01,

    logging_steps=100,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,

    remove_unused_columns=False,
    report_to="none",

    use_cpu=not torch.cuda.is_available()
)

# ==========================================
# STEP 10: TRAIN
# ==========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(
        tokenizer=tokenizer
    ),
    compute_metrics=compute_metrics
)

print("\nStarting improved fine-tuning...")

trainer.train()

# ==========================================
# STEP 11: EVALUATION
# ==========================================

print("\nEvaluating improved model...")

results = trainer.evaluate()

for name, value in results.items():
    print(f"{name}: {value}")

# ==========================================
# STEP 12: SAVE MODEL
# ==========================================

print("\nSaving improved model...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ==========================================
# STEP 13: CLASSIFICATION REPORT
# ==========================================

output = trainer.predict(val_dataset)

predictions = np.argmax(
    output.predictions,
    axis=-1
)

actual = output.label_ids

report = classification_report(
    actual,
    predictions,
    labels=[0, 1, 2],
    target_names=["Negative", "Neutral", "Positive"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

with open(
    os.path.join(REPORT_DIR, "improved_classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# ==========================================
# STEP 14: CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    actual,
    predictions,
    labels=[0, 1, 2]
)

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual Negative", "Actual Neutral", "Actual Positive"],
    columns=[
        "Predicted Negative",
        "Predicted Neutral",
        "Predicted Positive"
    ]
)

matrix_df.to_csv(
    os.path.join(REPORT_DIR, "improved_confusion_matrix.csv")
)

pd.DataFrame([results]).to_csv(
    os.path.join(REPORT_DIR, "improved_evaluation_results.csv"),
    index=False
)

# ==========================================
# COMPLETION
# ==========================================

print("\n" + "=" * 60)
print("IMPROVED MODEL TRAINING COMPLETED!")
print("=" * 60)

print("Model:", OUTPUT_DIR)
print("Metrics: reports/improved_evaluation_results.csv")
print("Report: reports/improved_classification_report.txt")
print("Confusion matrix: reports/improved_confusion_matrix.csv")
