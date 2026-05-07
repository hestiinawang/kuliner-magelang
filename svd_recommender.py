import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class SVDRecommender:
    def __init__(self, n_factors=10, n_iterations=50):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.user_idx = None
        self.item_idx = None
        self.predicted_ratings = None
    
    def fit(self, df):
        """
        Train SVD model dari user-item rating dataframe
        """
        # Create user-item matrix
        self.user_ids = df['user_id'].unique()
        self.item_ids = df['item'].unique()
        
        self.user_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_idx = {item: i for i, item in enumerate(self.item_ids)}
        
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        # Build sparse matrix
        row = [self.user_idx[u] for u in df['user_id']]
        col = [self.item_idx[i] for i in df['item']]
        data = df['rating'].values
        
        self.user_item_matrix = csr_matrix((data, (row, col)), shape=(n_users, n_items))
        
        # Fill missing with column mean (item average)
        matrix_dense = self.user_item_matrix.toarray()
        
        # Replace 0 with column mean (for unrated items)
        col_means = np.zeros(n_items)
        for j in range(n_items):
            col_data = matrix_dense[:, j]
            nonzero = col_data[col_data > 0]
            col_means[j] = np.mean(nonzero) if len(nonzero) > 0 else 0
        
        # Replace 0s with column means
        for i in range(n_users):
            for j in range(n_items):
                if matrix_dense[i, j] == 0:
                    matrix_dense[i, j] = col_means[j]
        
        # SVD decomposition
        k = min(self.n_factors, min(n_users, n_items) - 1)
        self.U, self.sigma, self.Vt = svds(csr_matrix(matrix_dense), k=k)
        
        # Reconstruct predicted ratings
        self.predicted_ratings = np.dot(np.dot(self.U, np.diag(self.sigma)), self.Vt)
        
        return self
    
    def recommend(self, user_id, n=5, exclude_rated=True):
        """
        Rekomendasikan top-N item untuk user
        """
        if user_id not in self.user_idx:
            return []
        
        user_idx = self.user_idx[user_id]
        user_predictions = self.predicted_ratings[user_idx, :]
        
        # Get items already rated
        rated_items = set()
        if exclude_rated:
            df_user = self.user_item_matrix.toarray()[user_idx, :]
            rated_items = {self.item_ids[i] for i in range(len(self.item_ids)) if df_user[i] > 0}
        
        # Create recommendation list
        recommendations = []
        for i, pred in enumerate(user_predictions):
            item = self.item_ids[i]
            if exclude_rated and item in rated_items:
                continue
            recommendations.append((item, pred))
        
        # Sort by predicted rating (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n]
    
    def predict_single(self, user_id, item):
        """
        Prediksi rating untuk user-item spesifik
        """
        if user_id not in self.user_idx or item not in self.item_idx:
            return None
        
        u_idx = self.user_idx[user_id]
        i_idx = self.item_idx[item]
        
        return round(self.predicted_ratings[u_idx, i_idx], 2)


def load_model():
    """Load data dan training model"""
    df = pd.read_csv('user_item_matrix.csv')
    model = SVDRecommender(n_factors=15, n_iterations=50)
    model.fit(df)
    return model


def get_all_users():
    """Get list semua user"""
    df = pd.read_csv('user_item_matrix.csv')
    return sorted(df['user_id'].unique())


if __name__ == "__main__":
    # Test
    model = load_model()
    users = get_all_users()
    
    print("=== SVD Recommender Test ===")
    print(f"Users: {len(users)}")
    print(f"Items: {len(model.item_ids)}")
    
    # Test rekomendasi untuk U001
    print("\nRekomendasi untuk U001:")
    recs = model.recommend('U001', n=10)
    for item, pred in recs:
        print(f"  {item}: {pred:.2f}")