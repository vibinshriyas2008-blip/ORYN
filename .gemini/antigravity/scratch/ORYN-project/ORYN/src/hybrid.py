import pandas as pd
from .collaborative import CollaborativeFilter
from .content_based import ContentBasedFilter
from .popularity import PopularityRecommender
from .cold_start import handle_cold_start
from .preprocessing import get_interaction_matrix

def get_hybrid_weights(interaction_count):
    if interaction_count == 0:
        return {'cf': 0.0, 'cb': 1.0}
    elif interaction_count <= 5:
        return {'cf': 0.2, 'cb': 0.8}
    elif interaction_count <= 15:
        return {'cf': 0.4, 'cb': 0.6}
    elif interaction_count <= 50:
        return {'cf': 0.65, 'cb': 0.35}
    else:
        return {'cf': 0.8, 'cb': 0.2}

class HybridRecommender:
    def __init__(self):
        self.cf = CollaborativeFilter(n_components=50)
        self.cb = ContentBasedFilter()
        self.pop = PopularityRecommender()
        self.user_interaction_counts = {}
        self.user_history = {}

    def fit(self, ratings_df, movies_df, tags_df=None):
        print("Training Popularity...")
        self.pop.fit(ratings_df)
        
        print("Training Content-Based...")
        self.cb.fit(movies_df, tags_df)
        
        print("Training Collaborative...")
        matrix, uid_map, mid_map = get_interaction_matrix(ratings_df)
        self.cf.fit(matrix, uid_map, mid_map)
        
        self.user_interaction_counts = ratings_df.groupby('userId').size().to_dict()
        self.user_history = ratings_df.groupby('userId')['movieId'].apply(set).to_dict()

    def recommend(self, user_id, interaction_count=None, n=20, user_prefs=None):
        if interaction_count is None:
            interaction_count = self.user_interaction_counts.get(user_id, 0)
            
        user_prefs = user_prefs or {}
        liked_movies = list(self.user_history.get(user_id, set()))
        if not liked_movies and 'liked_movie_ids' in user_prefs:
            liked_movies = user_prefs['liked_movie_ids']
            
        user_prefs['liked_movie_ids'] = liked_movies
            
        if interaction_count <= 5:
            recs = handle_cold_start(user_prefs, interaction_count, self.pop, self.cb, n=n)
            # Format outputs
            formatted_recs = []
            for m_id, score in recs:
                exp = 'Popular with users like you' if interaction_count == 0 else 'Because you liked similar movies'
                formatted_recs.append({'movie_id': m_id, 'score': float(score), 'cf_score': 0.0, 'cb_score': float(score), 'explanation': exp})
            return formatted_recs
            
        weights = get_hybrid_weights(interaction_count)
        
        exclude_ids = self.user_history.get(user_id, set())
        
        # Get CF candidates
        cf_recs = dict(self.cf.recommend(user_id, n=n*2, exclude_ids=exclude_ids))
        
        # Get CB candidates
        cb_recs = dict(self.cb.recommend_from_profile(liked_movies[-20:], n=n*2, exclude_ids=exclude_ids))
        
        all_candidates = set(cf_recs.keys()).union(set(cb_recs.keys()))
        
        final_scores = []
        for m_id in all_candidates:
            cf_score = cf_recs.get(m_id, 0)
            cb_score = cb_recs.get(m_id, 0)
            
            # Normalize CF score roughly (it can be outside [0,1], let's assume it's scaled)
            # CB score is cosine similarity [0,1]. Just linearly combine.
            final_score = weights['cf'] * cf_score + weights['cb'] * cb_score
            
            # Explanation
            if cf_score > cb_score and cf_score > 0.5:
                exp = 'Hidden gem matching your profile'
            elif cb_score > cf_score and cb_score > 0.5:
                exp = 'Matches your taste'
            else:
                exp = 'Trending in your taste zone'
                
            final_scores.append({
                'movie_id': m_id, 
                'score': float(final_score), 
                'cf_score': float(cf_score), 
                'cb_score': float(cb_score), 
                'explanation': exp
            })
            
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        return final_scores[:n]
