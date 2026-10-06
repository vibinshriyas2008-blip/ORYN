import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def split_train_test(ratings_df, test_ratio=0.2):
    # Sort by timestamp
    ratings_sorted = ratings_df.sort_values('timestamp')
    
    # Train test split (time-based)
    split_idx = int(len(ratings_sorted) * (1 - test_ratio))
    train_df = ratings_sorted.iloc[:split_idx].copy()
    test_df = ratings_sorted.iloc[split_idx:].copy()
    
    return train_df, test_df

def get_interaction_matrix(ratings_df):
    """
    Creates a sparse user-item interaction matrix.
    Returns: matrix (CSR), user_id_map, movie_id_map
    """
    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()
    
    user_id_map = {u: i for i, u in enumerate(unique_users)}
    movie_id_map = {m: i for i, m in enumerate(unique_movies)}
    
    row = ratings_df['userId'].map(user_id_map).values
    col = ratings_df['movieId'].map(movie_id_map).values
    data = ratings_df['rating'].values
    
    matrix = coo_matrix((data, (row, col)), shape=(len(unique_users), len(unique_movies))).tocsr()
    
    return matrix, user_id_map, movie_id_map

def get_tfidf_matrix(movies_df, tags_df=None):
    if tags_df is not None:
        tags_grouped = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
        movies = pd.merge(movies_df, tags_grouped, on='movieId', how='left')
        movies['tag'] = movies['tag'].fillna('')
        movies['combined'] = movies['genres'].str.replace('|', ' ') + ' ' + movies['tag']
    else:
        movies = movies_df.copy()
        movies['combined'] = movies['genres'].str.replace('|', ' ')
        
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined'])
    return tfidf_matrix, movies, tfidf
