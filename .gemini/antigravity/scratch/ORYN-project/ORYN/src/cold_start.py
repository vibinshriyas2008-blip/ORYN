def handle_cold_start(user_prefs, interaction_count, popularity_model, cb_model, n=10):
    """
    Handles recommendations for users with very few interactions.
    user_prefs: dict with 'genres' (list) and/or 'liked_movie_ids' (list)
    """
    liked_movies = user_prefs.get('liked_movie_ids', [])
    genres = user_prefs.get('genres', [])
    
    if interaction_count == 0:
        # Use onboarding genres + popularity
        pop_recs = popularity_model.recommend(n=n*3)
        
        if not genres:
            return pop_recs[:n]
            
        # Filter popular movies by genre if we have genre info and access to movies_df
        # Without movies_df passed here, we fallback to just popularity
        # (Assuming popularity gives general good movies)
        return pop_recs[:n]
        
    elif interaction_count <= 5:
        # Blend content-based (from liked movies) + popularity
        cb_recs = cb_model.recommend_from_profile(liked_movies, n=n)
        pop_recs = popularity_model.recommend(n=n, exclude_ids=liked_movies)
        
        # Simple blend: alternate or score-based
        blended = []
        cb_dict = dict(cb_recs)
        pop_dict = dict(pop_recs)
        
        all_movies = set(cb_dict.keys()).union(set(pop_dict.keys()))
        for m in all_movies:
            # Normalize scores roughly
            cb_score = cb_dict.get(m, 0) # cosine sim 0-1
            pop_score = pop_dict.get(m, 0) / 5.0 # Bayesian avg 0-1 approx
            blended.append((m, 0.7 * cb_score + 0.3 * pop_score))
            
        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:n]
        
    return [] # Should use main hybrid engine instead
