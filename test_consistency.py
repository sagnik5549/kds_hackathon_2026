from src.chunking import chunk_text
from src.embeddings import embed_texts
from src.consistency import decide_consistency


def main():
    # Load novel
    with open(
        "data/books/The Count of Monte Cristo.txt",
        encoding="utf-8"
    ) as f:
        text = f.read()

    # Prepare data
    chunks = chunk_text(text)
    chunks_small = chunks[:600]   # enough context
    embeddings = embed_texts(chunks_small)

    # Test backstory
    backstory = """
    Edmond Dantès is falsely imprisoned in the Château d'If
    and later escapes to seek revenge against those who betrayed him.
    """

    decision, evidence = decide_consistency(
        backstory,
        chunks_small,
        embeddings,
        threshold=0.6,
        k=5,
    )

    # Output
    print("=" * 60)
    print("BACKSTORY CONSISTENCY RESULT")
    print("=" * 60)
    print("Decision:", decision)
    print("\nTop Evidence:\n")

    for i, e in enumerate(evidence, 1):
        print(f"[Evidence {i}]  Score: {round(e['score'], 3)}")
        print(e["text"][:300])
        print("-" * 60)


if __name__ == "__main__":
    main()
