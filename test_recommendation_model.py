import pandas as pd
import numpy as np
from data_preprocessing import load_data, preprocess_data
from recommendation_model import build_collaborative_filtering_model, get_top_n_recommendations

# Path to the sample dataset
filepath = 'D:/Coding/AI_PRSys/data/user_interactions.csv'

# Load and preprocess the data
data = load_data(filepath)
preprocessed_data = preprocess_data(data)

# Create the user-item interaction matrix
user_item_matrix = preprocessed_data.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)

# Build the collaborative filtering model
W, H = build_collaborative_filtering_model(user_item_matrix)

# Generate predicted ratings
predicted_ratings = np.dot(W, H)

# Get recommendations for a specific user
user_id = 1
recommended_products = get_top_n_recommendations(user_id, user_item_matrix, predicted_ratings)

print("\nRecommended Products for User:", user_id)
print(recommended_products)