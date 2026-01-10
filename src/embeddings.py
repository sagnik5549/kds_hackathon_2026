from sentence_transformers import SentenceTransformer

# Lightweight and strong for semantic retrieval
MODEL_NAME = "all-MiniLM-L6-v2"

def embed_texts(texts):
    """
    Converts list of texts into embeddings.

    Args:
        texts (List[str]): Text chunks

    Returns:
        numpy.ndarray: Embeddings
    """
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings
