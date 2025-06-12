from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import numpy as np
from advanced_models import (
    NeuralCollaborativeFiltering,
    ContentBasedRecommender,
    KnowledgeGraphRecommender,
    TimeAwareRecommender,
    HybridRecommender
)
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize recommendation models
hybrid_recommender = HybridRecommender()

# Load and preprocess data
def load_data():
    """Load all necessary data for the recommendation system."""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Load user interactions
        interactions = pd.read_csv(os.path.join(BASE_DIR, 'data', 'user_interactions.csv'))
        
        # Load product data
        products = pd.read_csv(os.path.join(BASE_DIR, 'data', 'products.csv'))
        
        # Load user data
        users = pd.read_csv(os.path.join(BASE_DIR, 'data', 'users.csv'))
        
        # Build knowledge graph
        hybrid_recommender.kg_recommender.build_graph(
            products.to_dict('records'),
            users.to_dict('records'),
            interactions.to_dict('records')
        )
        
        # Analyze time patterns
        hybrid_recommender.time_recommender.analyze_time_patterns(
            interactions.to_dict('records')
        )
        
        # Train NCF model
        hybrid_recommender.train_ncf(interactions)
        
        print("Data loaded and NCF model trained successfully")
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

# Load data on startup
load_data()

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Enhanced recommendation endpoint that uses multiple AI models.
    
    Expects a JSON payload with:
    - user_id: ID of the user
    - context: Dictionary containing user context (time, location, etc.)
    - top_n: Number of recommendations to return (default: 5)
    
    Returns:
        JSON response with recommended products and their scores.
    """
    try:
        # Parse the incoming JSON request
        data = request.json
        user_id = data.get('user_id')
        context = data.get('context', {})
        top_n = data.get('top_n', 5)
        
        if user_id is None:
            return jsonify({'error': 'Missing user_id in request'}), 400
            
        # Add current time to context if not provided
        if 'current_time' not in context:
            context['current_time'] = datetime.now()
        
        # Get recommendations from hybrid model
        recommendations = hybrid_recommender.get_recommendations(
            user_id=user_id,
            user_context=context,
            top_n=top_n
        )
        
        # Format recommendations
        formatted_recommendations = [
            {
                'product_id': product_id,
                'score': float(score),
                'explanation': get_recommendation_explanation(product_id, score)
            }
            for product_id, score in recommendations
        ]
        
        return jsonify({
            'user_id': user_id,
            'recommendations': formatted_recommendations,
            'context_used': context
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_recommendation_explanation(product_id, score):
    """Generate a human-readable explanation for the recommendation."""
    return f"Recommended based on your preferences and behavior patterns (confidence: {score:.2f})"

@app.route('/train', methods=['POST'])
def train_models():
    """
    Endpoint to retrain the recommendation models with new data.
    """
    try:
        success = load_data()
        if success:
            return jsonify({'message': 'Models retrained successfully'})
        else:
            return jsonify({'error': 'Failed to retrain models'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    