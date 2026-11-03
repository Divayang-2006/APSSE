import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("qdrant_store")

QDRANT_AVAILABLE = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except Exception as e:
    logger.warning(f"qdrant-client not available: {e}")

STORAGE_DIR = Path(__file__).resolve().parent / "vector_db_data"

class QdrantVectorStore:
    """Standalone Vector Database client using Qdrant (distributed server or embedded storage)."""

    def __init__(self, host: Optional[str] = None, port: int = 6333):
        self.client = None
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", str(port)))
        self.use_server = False
        self._init_client()

    def _init_client(self):
        if not QDRANT_AVAILABLE:
            logger.error("Qdrant client library is not installed.")
            return

        # Attempt connection to standalone Qdrant server first
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            res = sock.connect_ex((self.host, self.port))
            sock.close()
            if res == 0:
                self.client = QdrantClient(host=self.host, port=self.port, timeout=5.0)
                self.use_server = True
                logger.info(f"Connected to Qdrant server at {self.host}:{self.port}")
                return
        except Exception:
            pass

        # Embedded persistent disk storage mode
        try:
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(STORAGE_DIR))
            self.use_server = False
            logger.info(f"Initialized embedded persistent Qdrant at {STORAGE_DIR}")
        except Exception as e:
            logger.warning(f"Could not initialize local path Qdrant ({e}), using in-memory mode.")
            self.client = QdrantClient(":memory:")

    def ensure_collection(self, collection_name: str, dimension: int):
        """Ensures the collection exists with the specified dimension and Cosine metric."""
        if self.client is None:
            return
        try:
            collections_resp = self.client.get_collections()
            existing_names = [col.name for col in collections_resp.collections]
            if collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection '{collection_name}' (dim={dimension}, metric=Cosine)")
        except Exception as e:
            logger.error(f"Error ensuring collection '{collection_name}': {e}")

    def upsert_batch(self, collection_name: str, points: List[PointStruct], batch_size: int = 200):
        """Upserts a list of PointStruct points in batches."""
        if self.client is None or not points:
            return 0
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            chunk = points[i:i + batch_size]
            try:
                self.client.upsert(collection_name=collection_name, points=chunk)
                total_upserted += len(chunk)
            except Exception as e:
                logger.error(f"Failed to upsert batch {i}-{i+len(chunk)} into {collection_name}: {e}")
        return total_upserted

    def search(self, collection_name: str, query_vector: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """Performs cosine vector search and returns matched payloads with similarity scores."""
        if self.client is None:
            return []

        q = query_vector.astype(np.float32).tolist()
        try:
            if hasattr(self.client, "query_points"):
                res = self.client.query_points(
                    collection_name=collection_name,
                    query=q,
                    limit=top_k,
                    with_payload=True
                )
                points = res.points
            else:
                points = self.client.search(
                    collection_name=collection_name,
                    query_vector=q,
                    limit=top_k,
                    with_payload=True
                )

            results = []
            for hit in points:
                payload = dict(hit.payload) if hit.payload else {}
                score = round(float(hit.score), 2)
                # Cosine similarity in Qdrant is between -1 and 1, clipped to 0..1 for UI display
                score = max(0.0, min(1.0, score))
                payload["similarity"] = score
                payload["qdrant_id"] = str(hit.id)
                results.append(payload)
            return results
        except Exception as e:
            logger.error(f"Search query error in Qdrant collection '{collection_name}': {e}")
            return []

    def get_case(self, collection_name: str, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single case by case_id string from payload."""
        if self.client is None:
            return None
        try:
            flt = Filter(must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))])
            res, _ = self.client.scroll(collection_name=collection_name, scroll_filter=flt, limit=1, with_payload=True)
            if res:
                return res[0].payload
        except Exception as e:
            logger.error(f"Failed to fetch case '{case_id}': {e}")
        return None

    def get_all_cases(self, collection_name: str, limit: int = 50, damage_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves cases from Qdrant with optional filtering."""
        if self.client is None:
            return []
        try:
            flt = None
            if damage_type:
                flt = Filter(must=[FieldCondition(key="damage_type", match=MatchValue(value=damage_type))])
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=flt,
                limit=limit,
                with_payload=True
            )
            return [r.payload for r in records if r.payload]
        except Exception as e:
            logger.error(f"Error scrolling cases: {e}")
            return []

    def count(self, collection_name: str) -> int:
        """Returns total vector count in collection."""
        if self.client is None:
            return 0
        try:
            res = self.client.count(collection_name=collection_name, exact=True)
            return res.count
        except Exception:
            return 0

# Global singleton
qdrant_store = QdrantVectorStore()

