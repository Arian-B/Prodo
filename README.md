# Prodo - Advanced AI Product Recommendation System

A sophisticated product recommendation system that combines multiple AI approaches for personalized recommendations.

## Features

- Neural Collaborative Filtering
- Content-based recommendations using BERT and CNN
- Graph-based recommendations
- Time-aware recommendations
- Hybrid recommendation approach

## Project Structure

```
Prodo/
├── advanced_models.py    # Core AI models implementation
├── app.py               # Flask API endpoints
├── requirements.txt     # Project dependencies
├── data/               # Data directory
│   ├── products.csv    # Product information
│   ├── users.csv       # User information
│   └── user_interactions.csv  # User interaction data
├── models/             # Directory for trained models
└── utils.py            # Utility functions
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

3. Run the application:
```bash
python app.py
```

## API Endpoints

- POST `/recommend`: Get personalized recommendations
- POST `/train`: Retrain the recommendation models

## Data Format

- `products.csv`: Product information (id, name, description, category, price, image_url)
- `users.csv`: User information (id, name, age, preferences, location)
- `user_interactions.csv`: User interaction data (user_id, product_id, clicks, rating) 