import pandas as pd
import numpy as np

from src.io_utils import load_book
from src.chunking import chunk_text
from src.embeddings import embed_texts
from src.consistency import decide_consistency



# Helpers

def label_to_int(label: str) -> int:
    """
    Convert CSV label to int
    consistent   -> 1
    contradict   -> 0
    """
    return 1 if label.lower().strip() == "consistent" else 0



# Load training data

df = pd.read_csv("data/train.csv")

scores = []
labels = []

print(f"Loaded {len(df)} training samples\n")


# Main loop

for _, row in df.iterrows():
    claim = row["content"]
    book = row["book_name"]
    true_label = label_to_int(row["label"])

    # Load and process book
    text = load_book(book)
    chunks = chunk_text(text)[:200]          # limit for speed
    embeddings = embed_texts(chunks)

    # Model prediction
    pred, evidence = decide_consistency(
        claim,
        chunks,
        embeddings,
        threshold=0.0  # TEMP — threshold applied later
    )

    best_score = evidence[0]["score"]

    scores.append(best_score)
    labels.append(true_label)


print("\nThreshold sweep:")
best_acc = 0.0
best_threshold = 0.0

for t in np.arange(0.45, 0.75, 0.01):
    preds = [1 if s >= t else 0 for s in scores]
    acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)

    print(f"Threshold {t:.2f} → Accuracy {acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        best_threshold = t


print("TRAINING COMPLETE")
print(f"Best Threshold : {best_threshold:.2f}")
print(f"Best Accuracy  : {best_acc:.4f}")
