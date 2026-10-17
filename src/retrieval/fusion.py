"""Reciprocal Rank Fusion (RRF) and candidate deduplication for hybrid retrieval."""

from typing import Dict, List, Optional
from src.retrieval.schemas import CandidateChunk


def reciprocal_rank_fusion(
    dense_candidates: List[CandidateChunk],
    sparse_candidates: List[CandidateChunk],
    k: int = 60,
    top_n: int = 50,
) -> List[CandidateChunk]:
    """Combine ranked results from dense and sparse retrieval using Reciprocal Rank Fusion.

    Formula:
        RRF_Score(d) = sum_{m in {dense, sparse}} (1 / (k + rank_m(d)))

    Args:
        dense_candidates: Ranked candidate chunks from dense retrieval (1-indexed).
        sparse_candidates: Ranked candidate chunks from sparse BM25 retrieval (1-indexed).
        k: Smoothing constant to control relative weighting of top ranks (default: 60).
        top_n: Maximum number of fused candidates to return for subsequent reranking.

    Returns:
        List of deduplicated CandidateChunk sorted by descending RRF score.
    """
    fused_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}
    chunk_map: Dict[str, CandidateChunk] = {}

    # 1. Process dense candidates
    for rank, candidate in enumerate(dense_candidates, start=1):
        cid = candidate.chunk_id
        dense_ranks[cid] = rank
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + rank))
        if cid not in chunk_map or len(candidate.text) > len(chunk_map[cid].text):
            chunk_map[cid] = candidate

    # 2. Process sparse candidates
    for rank, candidate in enumerate(sparse_candidates, start=1):
        cid = candidate.chunk_id
        sparse_ranks[cid] = rank
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + rank))
        if cid not in chunk_map or len(candidate.text) > len(chunk_map[cid].text):
            chunk_map[cid] = candidate

    # 3. Sort by fused RRF score descending
    sorted_chunk_ids = sorted(
        fused_scores.keys(),
        key=lambda x: fused_scores[x],
        reverse=True,
    )

    fused_candidates: List[CandidateChunk] = []
    for final_rank, cid in enumerate(sorted_chunk_ids[:top_n], start=1):
        base = chunk_map[cid]
        # Create a clean fused copy with rank and score preservation
        fused_chunk = CandidateChunk(
            chunk_id=base.chunk_id,
            text=base.text,
            score=round(fused_scores[cid], 6),
            original_rank=final_rank,
            retrieval_route="hybrid",
            metadata=dict(base.metadata),
            dense_rank=dense_ranks.get(cid),
            sparse_rank=sparse_ranks.get(cid),
            rrf_score=round(fused_scores[cid], 6),
            rerank_score=None,
        )
        fused_candidates.append(fused_chunk)

    return fused_candidates
