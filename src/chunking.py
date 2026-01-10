def chunk_text(text, chunk_size=800, overlap=100):
    """
    Splits text into overlapping word chunks.

    Args:
        text (str): Full novel text
        chunk_size (int): Number of words per chunk
        overlap (int): Number of overlapping words

    Returns:
        List[str]: List of text chunks
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
