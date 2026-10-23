import logging
import numpy as np
from typing import List, Tuple, Dict, Any

logger = logging.getLogger("vector_search")

FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception as e:
    logger.warning(f"FAISS not loaded: {e}. Running NumPy MIPS engine.")

class VectorIndex:
    """ANN vector index supporting HNSW (via FAISS) and Cosine Similarity."""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.ids: List[str] = []
        self.matrix: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)
        self.faiss_index = None
        self._build_faiss()

    def _build_faiss(self):
        if FAISS_AVAILABLE:
            try:
                # HNSW with Inner Product (Cosine similarity on normalized vectors)
                M = 16
                self.faiss_index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
                self.faiss_index.hnsw.efConstruction = 200
                self.faiss_index.hnsw.efSearch = 64
                logger.info(f"Initialized HNSW index (dim={self.dimension})")
            except Exception as e:
                logger.warning(f"Failed to build FAISS index: {e}")
                self.faiss_index = None

    def add(self, case_id: str, vector: np.ndarray):
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        
        self.ids.append(case_id)
        if len(self.matrix) == 0:
            self.matrix = np.expand_dims(vec, axis=0)
        else:
            self.matrix = np.vstack([self.matrix, np.expand_dims(vec, axis=0)])

        if self.faiss_index is not None:
            self.faiss_index.add(np.expand_dims(vec, axis=0))

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Searches top-k similar vectors using Cosine Similarity."""
        if len(self.ids) == 0:
            return []

        q = query_vec.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            q = q / norm

        top_k = min(top_k, len(self.ids))

        # FAISS HNSW query
        if self.faiss_index is not None and self.faiss_index.ntotal == len(self.ids):
            try:
                distances, indices = self.faiss_index.search(np.expand_dims(q, axis=0), top_k)
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    if 0 <= idx < len(self.ids):
                        sim = float(np.clip(dist, 0.0, 1.0))
                        results.append((self.ids[idx], round(sim, 2)))
                return results
            except Exception as e:
                logger.warning(f"FAISS search fallback: {e}")

        # NumPy Dot Product Cosine search
        sims = np.dot(self.matrix, q)
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            sim = float(np.clip(sims[idx], 0.0, 1.0))
            results.append((self.ids[idx], round(sim, 2)))
        return results

    def count(self) -> int:
        return len(self.ids)
