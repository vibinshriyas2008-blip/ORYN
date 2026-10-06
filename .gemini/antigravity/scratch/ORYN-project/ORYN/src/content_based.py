import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class ContentBasedFilter:
    def __init__(self):
        self.tfidf_matrix = None
        self.movie_indices = {}
        self.movies_df = None
        
    def fit(self, movies_df, tags_df=None):
        self.movies_df = movies_df.copy()
        
        # Prepare text data: title + genres + tags
        if tags_df is not None and not tags_df.empty:
            tags_grouped = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
            df_merged = pd.merge(self.movies_df, tags_grouped, on='movieId', how='left')
            df_merged['tag'] = df_merged['tag'].fillna('')
        else:
            df_merged = self.movies_df.copy()
            df_merged['tag'] = ''
            
        # Replace pipes in genres
        df_merged['genres'] = df_merged['genres'].str.replace('|', ' ', regex=False)
        df_merged['combined'] = df_merged['title'] + ' ' + df_merged['genres'] + ' ' + df_merged['tag']
        
        self.movie_indices = {m_id: idx for idx, m_id in enumerate(df_merged['movieId'])}
        
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(df_merged['combined'])
        
    def recommend_from_movie(self, movie_id, n=10):
        if movie_id not in self.movie_indices:
            return []
            
        idx = self.movie_indices[movie_id]
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # Sort and get top n (excluding the movie itself)
        sim_indices = sim_scores.argsort()[::-1]
        
        recommendations = []
        for i in sim_indices:
            m_id = self.movies_df.iloc[i]['movieId']
            if m_id != movie_id:
                recommendations.append((m_id, sim_scores[i]))
            if len(recommendations) >= n:
                break
                
        return recommendations
        
    def recommend_from_profile(self, liked_movie_ids, n=10, exclude_ids=None):
        valid_ids = [m for m in liked_movie_ids if m in self.movie_indices]
        if not valid_ids:
            return []
            
        exclude_ids = set(exclude_ids or [])
        exclude_ids.update(valid_ids) # don't recommend what they already liked
        
        indices = [self.movie_indices[m] for m in valid_ids]
        profile_vector = np.asarray(self.tfidf_matrix[indices].mean(axis=0))
        
        sim_scores = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()
        sim_indices = sim_scores.argsort()[::-1]
        
        recommendations = []
        for i in sim_indices:
            m_id = self.movies_df.iloc[i]['movieId']
            if m_id not in exclude_ids:
                recommendations.append((m_id, sim_scores[i]))
            if len(recommendations) >= n:
                break
                
        return recommendations

    def save(self, path):
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path):
        return joblib.load(path)
