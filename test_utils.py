from recommendation_model import build_collaborative_filtering_model
from data_preprocessing import load_data, preprocess_data
from utils import save_model, load_model
import pandas as pd

# Path to the sample dataset
filepath = 'D:/Coding/AI_PRSys/data/user_interactions.csv'

# Load and preprocess the data
data = load_data(filepath)
preprocessed_data = preprocess_data(data)

# Create the user-item interaction matrix
user_item_matrix = preprocessed_data.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)

# Build the collaborative filtering model
W, H = build_collaborative_filtering_model(user_item_matrix)

# Save the model components (W and H) to files
save_model(W, 'models/user_feature_matrix.pkl')
save_model(H, 'models/item_feature_matrix.pkl')

# Load the saved models
loaded_W = load_model('models/user_feature_matrix.pkl')
loaded_H = load_model('models/item_feature_matrix.pkl')

# Check if the loaded models are the same as the original ones
if loaded_W is not None and loaded_H is not None:
    print("\nLoaded models successfully!")
    print("Original W shape:", W.shape)
    print("Loaded W shape:", loaded_W.shape)
    print("Original H shape:", H.shape)
    print("Loaded H shape:", loaded_H.shape)