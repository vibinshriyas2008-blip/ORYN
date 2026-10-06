"""
ORYN Model Training Script
Downloads data if needed, trains all models, evaluates, saves results.
"""
import os
import sys
import json
import joblib

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_ratings, load_movies, load_tags
from src.preprocessing import split_train_test, get_interaction_matrix
from src.collaborative import CollaborativeFilter
from src.content_based import ContentBasedFilter
from src.popularity import PopularityRecommender
from src.hybrid import HybridRecommender
from src.reranking import rerank
from src.evaluation import Evaluator


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    data_dir = os.path.join(base_dir, 'data', 'ml-latest-small')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Check data exists
    if not os.path.exists(os.path.join(data_dir, 'ratings.csv')):
        print("Data not found. Running download script...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(base_dir, 'scripts', 'download_data.py')], check=True)

    print("=" * 60)
    print("  ORYN — Training ML Models")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags()
    print(f"  Ratings: {len(ratings):,} | Movies: {len(movies):,} | Tags: {len(tags):,}")

    print("\n[2/6] Train/test split (time-based 80/20)...")
    train_df, test_df = split_train_test(ratings)
    print(f"  Train: {len(train_df):,} ratings | Test: {len(test_df):,} ratings")

    print("\n[3/6] Training Popularity model...")
    pop = PopularityRecommender()
    pop.fit(train_df)
    joblib.dump(pop, os.path.join(models_dir, 'popularity.pkl'))
    print("  [OK] Popularity model saved")

    print("\n[4/6] Training Collaborative Filtering (SVD, k=50)...")
    cf = CollaborativeFilter(n_components=50)
    matrix, uid_map, mid_map = get_interaction_matrix(train_df)
    cf.fit(matrix, uid_map, mid_map)
    joblib.dump(cf, os.path.join(models_dir, 'cf.pkl'))
    print(f"  [OK] CF model saved  [matrix: {matrix.shape[0]} users × {matrix.shape[1]} movies]")

    print("\n[5/6] Training Content-Based (TF-IDF + Cosine Similarity)...")
    cb = ContentBasedFilter()
    cb.fit(movies, tags)
    joblib.dump(cb, os.path.join(models_dir, 'cb.pkl'))
    print(f"  [OK] CB model saved  [{len(movies)} movies in TF-IDF space]")

    print("\n[6/6] Training Hybrid Recommender...")
    hybrid = HybridRecommender()
    hybrid.fit(train_df, movies, tags)
    joblib.dump(hybrid, os.path.join(models_dir, 'hybrid.pkl'))
    print("  [OK] Hybrid model saved")

    print("\n" + "=" * 60)
    print("  Evaluating all models (this may take a minute)...")
    print("=" * 60)

    evaluator = Evaluator()
    results = evaluator.evaluate_all(
        test_df=test_df,
        movies_df=movies,
        pop_model=pop,
        cf_model=cf,
        cb_model=cb,
        hybrid_model=hybrid,
        rerank_fn=rerank,
        n_users=200,
    )

    results_file = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to: {results_file}\n")
    print("=" * 60)
    print("  RESULTS — Overall (all users)")
    print("=" * 60)
    print(f"{'Model':<20} {'P@5':>8} {'P@10':>8} {'R@5':>8} {'R@10':>8} {'NDCG@5':>8} {'NDCG@10':>8}")
    print("-" * 72)
    for model, metrics in results['overall'].items():
        print(f"{model:<20} {metrics['P@5']:>8.4f} {metrics['P@10']:>8.4f} "
              f"{metrics['R@5']:>8.4f} {metrics['R@10']:>8.4f} "
              f"{metrics['NDCG@5']:>8.4f} {metrics['NDCG@10']:>8.4f}")

    print("\n" + "=" * 60)
    print("  RESULTS — Cold Users (≤5 ratings)")
    print("=" * 60)
    print(f"{'Model':<20} {'P@5':>8} {'P@10':>8} {'NDCG@5':>8}")
    print("-" * 46)
    for model, metrics in results['cold_users'].items():
        print(f"{model:<20} {metrics['P@5']:>8.4f} {metrics['P@10']:>8.4f} {metrics['NDCG@5']:>8.4f}")

    print("\n" + "=" * 60)
    print("  RESULTS — Warm Users (>5 ratings)")
    print("=" * 60)
    print(f"{'Model':<20} {'P@5':>8} {'P@10':>8} {'NDCG@5':>8}")
    print("-" * 46)
    for model, metrics in results['warm_users'].items():
        print(f"{model:<20} {metrics['P@5']:>8.4f} {metrics['P@10']:>8.4f} {metrics['NDCG@5']:>8.4f}")

    print("\n[DONE] Training and evaluation complete!")


if __name__ == '__main__':
    main()
