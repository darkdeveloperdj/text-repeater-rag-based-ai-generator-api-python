import faiss
import numpy as np

def create_vectorstore(embeddings: np.ndarray):
    """
    Creates a FAISS index using the given embeddings.

    Parameters:
      embeddings (np.ndarray): The numpy array of embeddings.

    Returns:
      index: The FAISS index created using L2 distance.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index
