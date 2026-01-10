import pandas as pd

from src.io_utils import load_book
from src.chunking import chunk_text
from src.embeddings import embed_texts
from src.consistency import decide_consistency

df = pd.read_csv("data/test.csv")
outputs = []

import pandas as pd

from src.io_utils import load_book
from src.chunking import chunk_text
from src.embeddings import embed_texts
from src.consistency import decide_consistency


BEST_THRESHOLD = 0.58   
TOP_K_EVIDENCE = 3


df = pd.read_csv("data/test.csv")

outputs = []

print(f"Loaded {len(df)} test samples\n")


for _, row in df.iterrows():
    claim = row["content"]
    book = row["book_name"]

    text = load_book(book)
    chunks = chunk_text(text)[:200]
    embeddings = embed_texts(chunks)

    decision, evidence = decide_consistency(
        claim,
        chunks,
        embeddings,
        threshold=BEST_THRESHOLD
    )

    outputs.append({
        "id": row["id"],
        "decision": decision,
        "evidence": evidence[:TOP_K_EVIDENCE]
    })



print("\n_____________________________")
print("TEST RESULTS")
print("______________________________")

for o in outputs:
    print(f"\nID: {o['id']}")
    print(f"Decision: {'CONSISTENT' if o['decision'] == 1 else 'CONTRADICT'}")
    print("Top Evidence:")

    for i, e in enumerate(o["evidence"], 1):
        print(f"\n  [{i}] Score: {e['score']:.3f}")
        print(e["text"][:300])


for _, row in df.iterrows():
    claim = row["content"]       
    book = row["book_name"]       

    text = load_book(book)
    chunks = chunk_text(text)
    chunks = chunks[:200]

    embeddings = embed_texts(chunks)

    decision, evidence = decide_consistency(claim, chunks, embeddings)

    outputs.append({
        "id": row["id"],
        "decision": decision,
        "evidence": evidence[:3]
    })

pd.DataFrame(outputs).to_csv(
    "results/results.csv",
    index=False
)

print("Saved results/results.csv")
