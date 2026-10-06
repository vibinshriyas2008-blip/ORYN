import numpy as np
import pandas as pd
import joblib

class PopularityRecommender:
    def __init__(self):
        self.movie_scores = {}
        self.sorted_movies = []
        
    def fit(self, ratings_df):
        # Calculate Bayesian average
        stats = ratings_df.groupby('movieId')['rating'].agg(['count', 'mean'])
        
        global_mean = ratings_df['rating'].mean()
        # 'm' is the minimum number of votes required to be listed
        # Let's use the 75th percentile of counts as m
        m = stats['count'].quantile(0.75) if len(stats) > 0 else 0
        
        def bayesian_avg(row):
            v = row['count']
            R = row['mean']
            return (v / (v + m)) * R + (m / (v + m)) * global_mean
            
        stats['score'] = stats.apply(bayesian_avg, axis=1)
        
        self.movie_scores = stats['score'].to_dict()
        
        # Sort movies by score
        self.sorted_movies = sorted(self.movie_scores.items(), key=lambda x: x[1], reverse=True)
        
    def recommend(self, user_id=None, n=10, exclude_ids=None):
        exclude_ids = set(exclude_ids or [])
        recommendations = []
        
        for m_id, score in self.sorted_movies:
            if m_id not in exclude_ids:
                recommendations.append((m_id, float(score)))
            if len(recommendations) >= n:
                break
                
        return recommendations

    def save(self, path):
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path):
        return joblib.load(path)
