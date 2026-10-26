import numpy as np


class VectorIndex:
    """An in-memory vector store that finds the documents closest to a query by cosine similarity."""

    def __init__(self):
        self.documents = []
        # One normalized vector per row, so cosine similarity is a single matrix multiplication
        self._vectors = None

    def __len__(self):
        return len(self.documents)

    @staticmethod
    def _normalize(vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Cannot index an all-zero vector")
        return vectors / norms

    def add_vector(self, vector, document):
        """Add one embedding and the document (any value, e.g. a dict with the chunk text) it represents."""
        self.add_vectors([vector], [document])

    def add_vectors(self, vectors, documents):
        """Add many embeddings at once, e.g. the list returned by a single Voyage embed call."""
        if len(vectors) != len(documents):
            raise ValueError(f"Got {len(vectors)} vectors but {len(documents)} documents")
        if not vectors:
            return

        new_vectors = self._normalize(vectors)
        if new_vectors.ndim != 2:
            raise ValueError("Each vector must be a flat list of numbers")
        if self._vectors is not None and new_vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError(
                f"Vectors have {new_vectors.shape[1]} dimensions but the index has {self._vectors.shape[1]}. "
                "Use the same embedding model for every vector."
            )

        self._vectors = new_vectors if self._vectors is None else np.vstack([self._vectors, new_vectors])
        self.documents.extend(documents)

    def search(self, query_vector, k=3):
        """Return up to `k` (document, distance) pairs, closest first. Distance is 1 - cosine similarity."""
        if self._vectors is None:
            return []

        query = self._normalize(query_vector)
        if query.shape != (self._vectors.shape[1],):
            raise ValueError(f"Query must be a flat vector with {self._vectors.shape[1]} dimensions")

        similarities = self._vectors @ query
        k = min(k, len(self.documents))
        # argpartition finds the top k without sorting everything; then sort just those k
        top = np.argpartition(-similarities, k - 1)[:k]
        top = top[np.argsort(-similarities[top])]

        return [(self.documents[i], float(1 - similarities[i])) for i in top]
