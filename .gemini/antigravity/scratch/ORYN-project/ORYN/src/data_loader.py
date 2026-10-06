import os
import pandas as pd

def get_data_dir():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', 'ml-latest-small')

def load_ratings():
    path = os.path.join(get_data_dir(), 'ratings.csv')
    df = pd.read_csv(path)
    return df

def load_movies():
    path = os.path.join(get_data_dir(), 'movies.csv')
    df = pd.read_csv(path)
    # clean genres
    df['genres'] = df['genres'].fillna('').str.replace('|', ' ')
    return df

def load_tags():
    path = os.path.join(get_data_dir(), 'tags.csv')
    df = pd.read_csv(path)
    df['tag'] = df['tag'].fillna('')
    return df

def load_links():
    path = os.path.join(get_data_dir(), 'links.csv')
    return pd.read_csv(path)

def get_movie_metadata():
    movies = load_movies()
    tags = load_tags()
    
    # group tags by movie
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    
    # merge
    metadata = pd.merge(movies, tags_grouped, on='movieId', how='left')
    metadata['tag'] = metadata['tag'].fillna('')
    metadata['combined_features'] = metadata['genres'] + ' ' + metadata['tag']
    return metadata
