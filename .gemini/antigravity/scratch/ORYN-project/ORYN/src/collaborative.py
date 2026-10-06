import numpy as np
from scipy.sparse.linalg import svds
import joblib

class CollaborativeFilter:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.U = None
        self.sigma = None
        self.Vt = None
        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_user_map = {}
        self.reverse_movie_map = {}
        self.global_mean = 0

    def fit(self, matrix, user_id_map, movie_id_map):
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.reverse_user_map = {v: k for k, v in user_id_map.items()}
        self.reverse_movie_map = {v: k for k, v in movie_id_map.items()}
        
        # Calculate mean for centering
        self.global_mean = matrix.data.mean() if matrix.nnz > 0 else 0
        
        # Convert to float and mean center (simplified)
        matrix_float = matrix.astype(float)
        
        # Number of singular values must be strictly less than the minimum dimension of the matrix
        k = min(self.n_components, min(matrix_float.shape) - 1)
        if k < 1:
            k = 1
            
        U, sigma, Vt = svds(matrix_float, k=k)
        
        # Sort singular values in descending order
        sort_indices = np.argsort(sigma)[::-1]
        self.sigma = np.diag(sigma[sort_indices])
        self.U = U[:, sort_indices]
        self.Vt = Vt[sort_indices, :]

    def predict_rating(self, user_id, movie_id):
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return self.global_mean
        
        u_idx = self.user_id_map[user_id]
        m_idx = self.movie_id_map[movie_id]
        
        user_vector = self.U[u_idx, :]
        item_vector = self.Vt[:, m_idx]
        
        return np.dot(np.dot(user_vector, self.sigma), item_vector)

    def recommend(self, user_id, n=10, exclude_ids=None):
        if user_id not in self.user_id_map:
            return [] # Cold start
            
        exclude_ids = exclude_ids or []
        u_idx = self.user_id_map[user_id]
        user_vector = self.U[u_idx, :]
        
        # Predict for all movies
        all_preds = np.dot(np.dot(user_vector, self.sigma), self.Vt)
        
        # Create list of (movie_id, score)
        candidates = []
        for m_idx, score in enumerate(all_preds):
            m_id = self.reverse_movie_map[m_idx]
            if m_id not in exclude_ids:
                candidates.append((m_id, score))
                
        # Sort and return top n
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def save(self, path):
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path):
        return joblib.load(path)
