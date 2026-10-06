"""
ORYN Re-ranking Module
Applies diversity/novelty re-ranking using Maximal Marginal Relevance (MMR).
Balances relevance with genre diversity to avoid repetitive recommendations.
"""
import numpy as np


def _get_genres(movie_id, movies_df):
    """Return set of genres for a movie."""
    rows = movies_df[movies_df['movieId'] == movie_id]
    if rows.empty:
        return set()
    genres_str = rows.iloc[0]['genres']
    # genres may be pipe-separated or space-separated
    return set(genres_str.replace('|', ' ').split())


def _get_popularity(movie_id, movies_df, pop_scores=None):
    """Return a novelty score (inverse popularity). Range [0, 1]."""
    if pop_scores and movie_id in pop_scores:
        # Invert: less popular = more novel
        max_pop = max(pop_scores.values()) if pop_scores else 1.0
        return 1.0 - (pop_scores[movie_id] / max_pop)
    return 0.5  # neutral if unknown


def rerank(candidates, movies_df, lambda_diversity=0.3, n=10, pop_scores=None):
    """
    Re-rank candidates using Maximal Marginal Relevance (MMR) for diversity.

    Args:
        candidates: list of (movie_id, score) tuples OR list of dicts with 'movie_id'/'score'
        movies_df: DataFrame with movieId, genres columns
        lambda_diversity: trade-off parameter [0=pure relevance, 1=pure diversity]
        n: number of items to return
        pop_scores: optional dict {movie_id: pop_score} for novelty boosting

    Returns:
        list of (movie_id, mmr_score) tuples
    """
    if not candidates:
        return []

    # Normalize input format to list of dicts
    normalized = []
    for c in candidates:
        if isinstance(c, dict):
            normalized.append({'movie_id': c['movie_id'], 'score': c['score']})
        else:
            normalized.append({'movie_id': c[0], 'score': float(c[1])})

    if not normalized:
        return []

    # Normalize relevance scores to [0, 1]
    max_score = max(c['score'] for c in normalized)
    min_score = min(c['score'] for c in normalized)
    score_range = max_score - min_score if max_score != min_score else 1.0

    for c in normalized:
        c['norm_score'] = (c['score'] - min_score) / score_range

    reranked = []
    selected_genres = set()
    unselected = list(normalized)

    while len(reranked) < n and unselected:
        best_mmr = -float('inf')
        best_idx = 0

        for i, candidate in enumerate(unselected):
            m_id = candidate['movie_id']
            relevance = candidate['norm_score']

            # Genre-based diversity penalty
            genres = _get_genres(m_id, movies_df)
            if selected_genres and genres:
                overlap = len(genres & selected_genres) / len(genres)
            else:
                overlap = 0.0

            # Novelty bonus
            novelty = _get_popularity(m_id, movies_df, pop_scores)
            novelty_weight = 0.05  # small novelty boost

            mmr = ((1 - lambda_diversity) * relevance
                   - lambda_diversity * overlap
                   + novelty_weight * novelty)

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        selected = unselected.pop(best_idx)
        reranked.append((selected['movie_id'], selected['score']))
        selected_genres |= _get_genres(selected['movie_id'], movies_df)

    return reranked
