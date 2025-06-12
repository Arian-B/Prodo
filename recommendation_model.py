import numpy as np
from sklearn.decomposition import NMF

def build_collaborative_filtering_model(user_item_matrix, n_components=10):
    """
    Build a collaborative filtering model using Non-Negative Matrix Factorization (NMF).
    
    Args:
        user_item_matrix (pd.DataFrame): User-item interaction matrix.
        n_components (int): Number of latent factors (default is 10).
    
    Returns:
        W (np.ndarray): User-feature matrix.
        H (np.ndarray): Item-feature matrix.
    """
    # Initialize the NMF model
    model = NMF(n_components=n_components, init='random', random_state=42)
    
    # Fit the model to the user-item matrix
    W = model.fit_transform(user_item_matrix)  # User-feature matrix
    H = model.components_                     # Item-feature matrix
    
    print("Collaborative filtering model built successfully.")
    return W, H

def get_top_n_recommendations(user_id, user_item_matrix, predicted_ratings, top_n=5):
    """
    Get the top N recommendations for a specific user.
    
    Args:
        user_id (int): ID of the user for whom recommendations are generated.
        user_item_matrix (pd.DataFrame): User-item interaction matrix.
        predicted_ratings (np.ndarray): Predicted ratings matrix (W.dot(H)).
        top_n (int): Number of recommendations to return (default is 5).
    
    Returns:
        list: List of recommended product IDs.
    """
    try:
        # Get the index of the user in the user-item matrix
        user_index = user_item_matrix.index.get_loc(user_id)
        
        # Get the predicted ratings for the user
        user_ratings = predicted_ratings[user_index]
        
        # Get the indices of the top N products with the highest predicted ratings
        top_n_indices = np.argsort(user_ratings)[-top_n:][::-1]
        
        # Map indices back to product IDs
        recommended_product_ids = user_item_matrix.columns[top_n_indices]
        
        print(f"Top {top_n} recommendations for user {user_id}: {list(recommended_product_ids)}")
        return list(recommended_product_ids)
    
    except KeyError:
        print(f"User ID {user_id} not found in the user-item matrix.")
        return []