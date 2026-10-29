"""
qdrant_client.py

Connection, collection lifecycle, and health checks for the local Qdrant
instance (Docker, localhost:6333). Vector upsert/search logic does NOT
live here — see vector_ops.py. This module only owns "is Qdrant
reachable, does the collection exist, what shape should it be" — the
questions monitor.py and ingestion setup need answered before any actual
data moves.
"""
from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def get_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    return QdrantClient(host=host, port=port)


def health_check(client: QdrantClient) -> bool:
    """True if Qdrant is reachable and responding."""
    try:
        client.get_collections()
        return True
    except Exception:
        return False


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine",
    recreate: bool = False,
) -> None:
    """
    Create the collection if it doesn't exist. vector_size must match
    whichever embedder produced the vectors you're storing  — mixing embedders in one collection
    errors out on the first mismatched upsert.

    recreate=True wipes and rebuilds the collection — use this when
    switching embedders, since vectors from a different model aren't
    comparable to each other even at the same dimensionality.
    """
    exists = client.collection_exists(collection_name)
    if exists and not recreate:
        return
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=vector_size, distance=distance),
    )


def collection_info(client: QdrantClient, collection_name: str) -> Optional[dict]:
    """
    Point count + vector config for a collection, or None if it doesn't
    exist. Meant for monitor.py / sanity checks after ingestion, not for
    the hot search path.
    """
    if not client.collection_exists(collection_name):
        return None
    info = client.get_collection(collection_name)
    return {
        "collection_name": collection_name,
        "points_count": info.points_count,
        "vector_size": info.config.params.vectors.size,
        "distance": info.config.params.vectors.distance,
    }
