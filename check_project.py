import os

print("\n========== PROJECT FILE CHECK ==========\n")

folders = [
    "models",
    "dataset",
    "reports"
]

for folder in folders:
    print(f"\n{folder}/")

    if os.path.exists(folder):
        for item in os.listdir(folder):
            print("  ", item)
    else:
        print("   Folder not found")

print("\n========== MODEL CHECK ==========\n")

model_paths = [
    "models/sentiment-lora",
    "models/qwen2.5-1.5b-sentiment-lora"
]

for path in model_paths:
    print(path, "->", os.path.exists(path))

print("\n========== DATASET CHECK ==========\n")

for file in ["train.csv", "validation.csv", "test.csv"]:
    path = os.path.join("dataset", file)
    print(file, "->", os.path.exists(path))

print("\n========== CHECK COMPLETED ==========")