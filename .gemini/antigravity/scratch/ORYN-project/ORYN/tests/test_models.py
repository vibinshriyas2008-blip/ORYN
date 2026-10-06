import pytest
import pandas as pd
import numpy as np

from src.preprocessing import split_train_test, get_interaction_matrix
from src.popularity import PopularityRecommender
from src.collaborative import CollaborativeFilter
from src.content_based import ContentBasedFilter
from src.hybrid import HybridRecommender
from src.evaluation import precision_at_k

@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        'userId': [1, 1, 2, 2, 3],
        'movieId': [10, 20, 10, 30, 20],
        'rating': [5.0, 4.0, 4.5, 3.0, 2.0],
        'timestamp': [100, 101, 102, 103, 104]
    })

@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        'movieId': [10, 20, 30],
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'genres': ['Action|Comedy', 'Comedy|Drama', 'Action|Sci-Fi']
    })

def test_preprocessing(sample_ratings):
    train, test = split_train_test(sample_ratings, test_ratio=0.2)
    assert len(train) == 4
    assert len(test) == 1
    
    matrix, u_map, m_map = get_interaction_matrix(sample_ratings)
    assert matrix.shape == (3, 3)

def test_popularity(sample_ratings):
    pop = PopularityRecommender()
    pop.fit(sample_ratings)
    recs = pop.recommend(n=2)
    assert len(recs) == 2
    # 10 has highest avg (4.75) and 2 counts, should be first
    assert recs[0][0] == 10

def test_collaborative(sample_ratings):
    cf = CollaborativeFilter(n_components=2)
    matrix, u_map, m_map = get_interaction_matrix(sample_ratings)
    cf.fit(matrix, u_map, m_map)
    recs = cf.recommend(1, n=2)
    assert isinstance(recs, list)

def test_content_based(sample_movies):
    cb = ContentBasedFilter()
    cb.fit(sample_movies)
    recs = cb.recommend_from_movie(10, n=2)
    assert len(recs) <= 2

def test_evaluation():
    recommended = [1, 2, 3, 4, 5]
    actual = {2, 3, 6}
    p = precision_at_k(recommended, actual, 5)
    assert p == 0.4
