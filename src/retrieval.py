import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_top_k_numpy(chunks, embeddings, query_embedding, k=5):
    """
    Windows-safe retrieval using cosine similarity.
    Mirrors Pathway vector search logic.
    """
    scores = [
        cosine_similarity(query_embedding, emb)
        for emb in embeddings
    ]

    top_indices = np.argsort(scores)[::-1][:k]

    results = [
        {
            "chunk_id": int(i),
            "score": float(scores[i]),
            "text": chunks[i]
        }
        for i in top_indices
    ]

    return results
