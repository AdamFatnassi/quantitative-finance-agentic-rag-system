"""
vector_ops.py

Upsert / delete / search operations against a Qdrant collection. Assumes
the collection already exists — call qdrant_client.ensure_collection()
first (the ingestion pipeline / ingest script is responsible for that,
not this module).
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def chunk_id_to_point_id(chunk_id: str) -> int:
    """
    Qdrant point IDs must be int or UUID. chunk_id is already a stable
    md5-derived string from the chunking pipeline, so hash it down to a
    stable positive int instead of a random UUID — this makes upserts
    idempotent: re-running ingestion with the same chunk_id overwrites the
    existing point instead of creating a duplicate.
    """
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16)


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunk_ids: List[str],
    vectors,
    payloads: List[dict],
    batch_size: int = 100,
) -> int:
    """
    chunk_ids[i], vectors[i] and payloads[i] must correspond to the same
    item. payloads are taken as-is (the caller decides what to keep) —
    this function's only job is turning them into stable-id PointStructs
    and upserting them in batches. Returns the number of points upserted.
    """
    vectors = np.asarray(vectors)
    assert len(chunk_ids) == len(vectors) == len(payloads), \
        "chunk_ids/vectors/payloads length mismatch"

    points = [
        qm.PointStruct(
            id=chunk_id_to_point_id(cid),
            vector=vec.tolist(),
            payload=payload,
        )
        for cid, vec, payload in zip(chunk_ids, vectors, payloads)
    ]

    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i:i + batch_size])

    return len(points)


def delete_by_decision_id(client: QdrantClient, collection_name: str, decision_id: str) -> None:
    """
    Delete every chunk belonging to one decision_id — e.g. before
    re-ingesting a corrected version of that document, so stale chunks
    from the old version don't linger alongside the new ones.
    """
    client.delete(
        collection_name=collection_name,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="decision_id", match=qm.MatchValue(value=decision_id))]
            )
        ),
    )


def search(
    client: QdrantClient,
    collection_name: str,
    query_vector,
    k: int = 5,
    query_filter: Optional[qm.Filter] = None,
    decision_id: Optional[str] = None,
):
    """
    query_filter lets a caller (e.g. retrieve.py) pass an arbitrary
    pre-built Qdrant filter (language, decision_category,
    document_format, combinations thereof...). decision_id stays as a
    convenience shortcut for the common "just this one decision" case --
    if both are given, decision_id's condition is folded into
    query_filter's `must` list rather than overriding it.
    """
    if decision_id:
        decision_cond = qm.FieldCondition(key="decision_id", match=qm.MatchValue(value=decision_id))
        if query_filter is None:
            query_filter = qm.Filter(must=[decision_cond])
        else:
            query_filter = qm.Filter(
                must=[*(query_filter.must or []), decision_cond],
                should=query_filter.should,
                must_not=query_filter.must_not,
            )

    vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else list(query_vector)

    # client.search() was removed in recent qdrant-client versions --
    # query_points() is the replacement. It returns a QueryResponse whose
    # .points is the list of ScoredPoint that .search() used to return
    # directly, so callers iterating over "hits" as ScoredPoint objects
    # (payload, score, id) don't need to change.
    response = client.query_points(
        collection_name=collection_name,
        query=vec,
        query_filter=query_filter,
        limit=k,
    )
    return response.points

def count(client: QdrantClient, collection_name: str) -> int:
    return client.count(collection_name).count
