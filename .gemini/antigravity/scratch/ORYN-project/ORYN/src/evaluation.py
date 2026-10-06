"""
ORYN Evaluation Module
Computes Precision@K, Recall@K, NDCG@K for all models.
Uses proper train/test split to avoid data leakage.
"""
import numpy as np
import json
import os


def precision_at_k(recommended, actual, k):
    """Precision@K: fraction of top-K recs that are relevant."""
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in actual)
    return hits / k


def recall_at_k(recommended, actual, k):
    """Recall@K: fraction of relevant items found in top-K recs."""
    if not actual:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in actual)
    return hits / len(actual)


def ndcg_at_k(recommended, actual, k):
    """NDCG@K: normalized discounted cumulative gain."""
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)
    # Ideal DCG: all relevant items at the top
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _compute_metrics(recs, actual, k_values=(5, 10)):
    """Compute P, R, NDCG for multiple K values."""
    result = {}
    for k in k_values:
        result[f'P@{k}'] = precision_at_k(recs, actual, k)
        result[f'R@{k}'] = recall_at_k(recs, actual, k)
        result[f'NDCG@{k}'] = ndcg_at_k(recs, actual, k)
    return result


class Evaluator:
    """
    Evaluates recommendation models using leave-out test interactions.
    
    Protocol:
    - Train models on train_df
    - For each user in test set, their test items are ground truth
    - Use training history to build CB user profile (no leakage)
    - Compute P@K, R@K, NDCG@K for K in {5, 10}
    """

    def evaluate_all(self, test_df, movies_df, pop_model, cf_model, cb_model,
                     hybrid_model, rerank_fn=None, n_users=200):
        """
        Evaluate all models on test set.
        
        Args:
            test_df: held-out ratings (must have userId, movieId columns)
            movies_df: full movies dataframe
            pop_model: PopularityRecommender (already fitted)
            cf_model: CollaborativeFilter (already fitted)
            cb_model: ContentBasedFilter (already fitted)
            hybrid_model: HybridRecommender (already fitted)
            rerank_fn: optional reranking function
            n_users: max users to evaluate (for speed)
        
        Returns:
            dict with metrics per model, plus cold/warm split
        """
        test_users = test_df['userId'].unique()
        if len(test_users) > n_users:
            np.random.seed(42)
            test_users = np.random.choice(test_users, n_users, replace=False)

        # Ground truth: test interactions per user
        actual_interactions = test_df.groupby('userId')['movieId'].apply(set).to_dict()

        # Training history per user (for CB profile, exclusion list)
        train_history = hybrid_model.user_history  # {user_id -> set of movie_ids}

        models = ['Popularity', 'CF', 'CB', 'Hybrid', 'Hybrid+Rerank']
        metric_keys = ['P@5', 'P@10', 'R@5', 'R@10', 'NDCG@5', 'NDCG@10']

        results_warm = {m: {k: [] for k in metric_keys} for m in models}
        results_cold = {m: {k: [] for k in metric_keys} for m in models}

        interaction_counts = hybrid_model.user_interaction_counts

        for u in test_users:
            actual = actual_interactions.get(u, set())
            if not actual:
                continue

            n_train = interaction_counts.get(u, 0)
            is_cold = n_train <= 5

            # Exclude already-seen training items
            exclude = train_history.get(u, set())

            # --- Popularity ---
            pop_recs = [x[0] for x in pop_model.recommend(n=10, exclude_ids=list(exclude))]

            # --- Collaborative ---
            cf_raw = cf_model.recommend(u, n=10, exclude_ids=exclude)
            cf_recs = [x[0] for x in cf_raw]

            # --- Content-Based (use training history as profile) ---
            liked = list(exclude)[:20] if exclude else []
            cb_raw = cb_model.recommend_from_profile(liked, n=10, exclude_ids=list(exclude))
            cb_recs = [x[0] for x in cb_raw]

            # --- Hybrid ---
            hybrid_out = hybrid_model.recommend(u, interaction_count=n_train, n=10)
            hyb_recs = [x['movie_id'] for x in hybrid_out] if hybrid_out else []

            # --- Hybrid + Rerank ---
            if rerank_fn is not None:
                hybrid_candidates = hybrid_model.recommend(u, interaction_count=n_train, n=30)
                candidates_list = [(x['movie_id'], x['score']) for x in hybrid_candidates]
                reranked = rerank_fn(candidates_list, movies_df, n=10)
                rerank_recs = [x[0] for x in reranked]
            else:
                rerank_recs = hyb_recs

            recs_map = {
                'Popularity': pop_recs,
                'CF': cf_recs,
                'CB': cb_recs,
                'Hybrid': hyb_recs,
                'Hybrid+Rerank': rerank_recs,
            }

            target = results_cold if is_cold else results_warm

            for model_name, recs in recs_map.items():
                m = _compute_metrics(recs, actual)
                for k_str, val in m.items():
                    target[model_name][k_str].append(val)

        def _avg(results_dict):
            return {
                model: {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
                for model, metrics in results_dict.items()
            }

        overall = {}
        for model in models:
            combined = {}
            for k_str in metric_keys:
                all_vals = results_warm[model][k_str] + results_cold[model][k_str]
                combined[k_str] = float(np.mean(all_vals)) if all_vals else 0.0
            overall[model] = combined

        return {
            'overall': overall,
            'warm_users': _avg(results_warm),
            'cold_users': _avg(results_cold),
        }
