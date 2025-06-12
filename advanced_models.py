import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import numpy as np
import networkx as nx
from datetime import datetime
import spacy
from PIL import Image
import cv2
import random

class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model that combines matrix factorization
    with a multi-layer neural network.
    """
    def __init__(self, num_users, num_items, num_factors=32, layers=[64, 32, 16, 8]):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)
        
        # Neural network layers
        self.layers = nn.ModuleList()
        input_size = num_factors * 2
        for layer_size in layers:
            self.layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size
        
        self.output_layer = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)
        
        # Concatenate embeddings
        x = torch.cat([user_embedding, item_embedding], dim=1)
        
        # Pass through neural network
        for layer in self.layers:
            x = torch.relu(layer(x))
        
        # Output layer
        x = self.output_layer(x)
        return self.sigmoid(x)

    def get_recommendations(self, user_id, top_n=5):
        # Dummy implementation: return top_n random product IDs with random scores
        product_ids = [101, 102, 103, 104, 105]  # Use your actual product IDs or load dynamically
        recs = random.sample(product_ids, min(top_n, len(product_ids)))
        return [(pid, random.uniform(0.5, 1.0)) for pid in recs]

    def train_model(self, interactions, user_id_map, product_id_map, epochs=10, lr=0.001, batch_size=16):
        # Prepare training data
        user_indices = [user_id_map[u] for u in interactions['user_id']]
        item_indices = [product_id_map[i] for i in interactions['product_id']]
        ratings = interactions['rating'].values.astype(float)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(user_indices, dtype=torch.long),
            torch.tensor(item_indices, dtype=torch.long),
            torch.tensor(ratings, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for user, item, rating in loader:
                user, item, rating = user.to(device), item.to(device), rating.to(device)
                optimizer.zero_grad()
                output = self(user, item).squeeze()
                loss = loss_fn(output, rating)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * user.size(0)
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        self.to('cpu')
        print("NCF training complete.")

class ContentBasedRecommender:
    """
    Content-based recommender using BERT for text understanding and
    CNN for image feature extraction.
    """
    def __init__(self):
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.nlp = spacy.load('en_core_web_sm')
        
        # CNN for image feature extraction
        self.cnn_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )

    def extract_text_features(self, text):
        """Extract features from product descriptions using BERT."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def extract_image_features(self, image_path):
        """Extract features from product images using CNN."""
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return self.cnn_model.predict(img_array)

    def get_recommendations(self, user_id, top_n=5):
        # Dummy implementation: return top_n random product IDs with random scores
        product_ids = [101, 102, 103, 104, 105]  # Use your actual product IDs or load dynamically
        recs = random.sample(product_ids, min(top_n, len(product_ids)))
        return [(pid, random.uniform(0.5, 1.0)) for pid in recs]

class KnowledgeGraphRecommender:
    """
    Knowledge Graph-based recommender using graph neural networks.
    """
    def __init__(self):
        self.graph = nx.Graph()
        
    def build_graph(self, products, users, interactions):
        """Build a knowledge graph from products, users, and their interactions."""
        # Add nodes
        for product in products:
            self.graph.add_node(product['product_id'], type='product', **product)
        for user in users:
            self.graph.add_node(user['user_id'], type='user', **user)
            
        # Add edges
        for interaction in interactions:
            edge_kwargs = {'weight': interaction['rating']}
            if 'timestamp' in interaction:
                edge_kwargs['timestamp'] = interaction['timestamp']
            self.graph.add_edge(
                interaction['user_id'],
                interaction['product_id'],
                **edge_kwargs
            )

    def get_recommendations(self, user_id, top_n=5):
        """Get recommendations using graph-based algorithms."""
        # Use Personalized PageRank for recommendations
        pagerank = nx.pagerank(self.graph, personalization={user_id: 1.0})
        
        # Filter for product nodes and sort by score
        product_scores = {
            node: score for node, score in pagerank.items()
            if self.graph.nodes[node]['type'] == 'product'
        }
        
        return sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

class TimeAwareRecommender:
    """
    Time-aware recommender that considers temporal patterns in user behavior.
    """
    def __init__(self):
        self.time_patterns = {}
        
    def analyze_time_patterns(self, user_interactions):
        """Analyze temporal patterns in user interactions."""
        for interaction in user_interactions:
            user_id = interaction['user_id']
            if 'timestamp' not in interaction:
                continue  # Skip if no timestamp
            timestamp = datetime.fromtimestamp(interaction['timestamp'])
            
            if user_id not in self.time_patterns:
                self.time_patterns[user_id] = {
                    'hour_of_day': [],
                    'day_of_week': [],
                    'month': []
                }
            
            self.time_patterns[user_id]['hour_of_day'].append(timestamp.hour)
            self.time_patterns[user_id]['day_of_week'].append(timestamp.weekday())
            self.time_patterns[user_id]['month'].append(timestamp.month)

    def get_time_based_recommendations(self, user_id, current_time):
        """Get recommendations based on user's time patterns."""
        if user_id not in self.time_patterns:
            return []
            
        patterns = self.time_patterns[user_id]
        current_hour = current_time.hour
        current_weekday = current_time.weekday()
        
        # Calculate time similarity scores
        hour_scores = np.exp(-0.1 * np.abs(np.array(patterns['hour_of_day']) - current_hour))
        weekday_scores = np.exp(-0.1 * np.abs(np.array(patterns['day_of_week']) - current_weekday))
        
        # Combine scores
        time_scores = hour_scores * weekday_scores
        return time_scores

class HybridRecommender:
    """
    Hybrid recommender that combines multiple recommendation approaches.
    """
    def __init__(self):
        self.ncf_model = NeuralCollaborativeFiltering(num_users=1000, num_items=1000)
        self.content_recommender = ContentBasedRecommender()
        self.kg_recommender = KnowledgeGraphRecommender()
        self.time_recommender = TimeAwareRecommender()
        
    def get_recommendations(self, user_id, user_context, top_n=5):
        """
        Get hybrid recommendations combining multiple approaches.
        
        Args:
            user_id: ID of the user
            user_context: Dictionary containing user context (time, location, etc.)
            top_n: Number of recommendations to return
        """
        # Get recommendations from each model
        ncf_recs = self.ncf_model.get_recommendations(user_id, top_n)
        content_recs = self.content_recommender.get_recommendations(user_id, top_n)
        kg_recs = self.kg_recommender.get_recommendations(user_id, top_n)
        time_recs = self.time_recommender.get_time_based_recommendations(
            user_id, 
            user_context['current_time']
        )
        
        # Combine and re-rank recommendations
        combined_scores = {}
        for rec_type, recs in [
            ('ncf', ncf_recs),
            ('content', content_recs),
            ('kg', kg_recs),
            ('time', time_recs)
        ]:
            for item_id, score in recs:
                if item_id not in combined_scores:
                    combined_scores[item_id] = {'scores': [], 'count': 0}
                combined_scores[item_id]['scores'].append(score)
                combined_scores[item_id]['count'] += 1
        
        # Calculate final scores using weighted average
        final_scores = {}
        for item_id, data in combined_scores.items():
            final_scores[item_id] = sum(data['scores']) / data['count']
        
        # Return top N recommendations
        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def train_ncf(self, interactions):
        # Map user and product IDs to indices
        user_ids = sorted(interactions['user_id'].unique())
        product_ids = sorted(interactions['product_id'].unique())
        user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        product_id_map = {pid: idx for idx, pid in enumerate(product_ids)}
        self.ncf_model = NeuralCollaborativeFiltering(num_users=len(user_ids), num_items=len(product_ids))
        self.ncf_model.train_model(interactions, user_id_map, product_id_map)
        self.user_id_map = user_id_map
        self.product_id_map = product_id_map
        self.reverse_product_id_map = {v: k for k, v in product_id_map.items()} 