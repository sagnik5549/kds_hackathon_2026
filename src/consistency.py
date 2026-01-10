import numpy as np
from src.retrieval import retrieve_top_k_numpy
from src.embeddings import embed_texts


def decide_consistency(
    backstory_text,
    chunks,
    embeddings,
    threshold=0.45,
    k=5,
):
    """
    Decide whether backstory is consistent with the novel.

    Returns:
        decision (int): 1 = consistent, 0 = inconsistent
        evidence (list): top-k retrieved chunks
    """
    backstory_embedding = embed_texts([backstory_text])[0]

    evidence = retrieve_top_k_numpy(
        chunks,
        embeddings,
        backstory_embedding,
        k=k,
    )

    best_score = evidence[0]["score"]
    decision = 1 if best_score >= threshold else 0
    return decision, evidence
