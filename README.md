# Prodo - Advanced AI Product Recommendation System

A sophisticated product recommendation system that combines multiple AI approaches for personalized recommendations, including real Neural Collaborative Filtering (NCF) training.

---

## Features

- **Neural Collaborative Filtering (NCF)**: Deep learning-based collaborative filtering using PyTorch, trained on your user interactions.
- **Content-based Recommendations**: Uses BERT for text and CNN for image features.
- **Graph-based Recommendations**: Knowledge graph with user-product interactions.
- **Time-aware Recommendations**: Considers temporal patterns in user behavior.
- **Hybrid Recommendation Approach**: Combines all the above for robust, explainable recommendations.
- **REST API**: Easily integrate with any frontend or tool (e.g., Postman).
- **Robust Data Handling**: Works with or without timestamps in your interaction data.

---

## Project Structure

```
prodo/
├── advanced_models.py         # Core AI models (NCF, content, graph, time-aware, hybrid)
├── app.py                    # Flask API endpoints
├── requirements.txt          # Project dependencies
├── data/                     # Data directory
│   ├── products.csv          # Product information
│   ├── users.csv             # User information
│   └── user_interactions.csv # User interaction data
├── models/                   # Directory for trained models (optional)
├── utils.py                  # Utility functions
├── test_utils.py             # Test scripts
├── test_recommendation_model.py
├── test_data_preprocessing.py
├── data_preprocessing.py
└── README.md
```

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd prodo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **(Optional) Download BERT weights and ResNet weights:**  
   These will be automatically downloaded the first time you run the app.

5. **Prepare your data:**  
   Place your CSV files in the `data/` directory. See below for format.

6. **Run the application:**
   ```bash
   python app.py
   ```
   The API will be available at `http://127.0.0.1:5000/`

---

## Data Format

### `products.csv`
| product_id | name              | description                        | category    | price  | image_url                      |
|------------|-------------------|------------------------------------|-------------|--------|-------------------------------|
| 101        | Smart Watch Pro   | Advanced fitness tracking smartwatch| Electronics | 199.99 | https://example.com/watch.jpg  |

### `users.csv`
| user_id | name         | age | preferences         | location   |
|---------|--------------|-----|---------------------|------------|
| 1       | John Doe     | 28  | Electronics,Gaming  | New York   |

### `user_interactions.csv`
| user_id | product_id | clicks | rating | (timestamp, optional) |
|---------|------------|--------|--------|-----------------------|
| 1       | 101        | 5      | 4      | 1718246400            |

- `timestamp` is optional. If present, enables time-aware recommendations.

---

## API Endpoints

### 1. **POST `/recommend`**
Get personalized recommendations for a user.

- **Request:**
  - **Headers:** `Content-Type: application/json`
  - **Body:**
    ```json
    {
      "user_id": 1,
      "context": {},
      "top_n": 5
    }
    ```
    - `user_id`: The user to recommend for.
    - `context`: (optional) Dictionary with context (e.g., time, location).
    - `top_n`: (optional) Number of recommendations (default: 5).

- **Response:**
    ```json
    {
      "user_id": 1,
      "recommendations": [
        {
          "product_id": 101,
          "score": 0.92,
          "explanation": "Recommended based on your preferences and behavior patterns (confidence: 0.92)"
        }
      ],
      "context_used": { ... }
    }
    ```

#### **How to use with Postman:**
1. Set method to `POST` and URL to `http://127.0.0.1:5000/recommend`
2. In the **Headers** tab, add:  
   - Key: `Content-Type`  
   - Value: `application/json`
3. In the **Body** tab, select `raw` and choose `JSON` from the dropdown.
4. Paste the example JSON above and click **Send**.

---

### 2. **POST `/train`**
Retrain all models (including NCF) on the latest data.

- **Request:** No body required.
- **Response:**
    ```json
    { "message": "Models retrained successfully" }
    ```

---

## Model Training

- The NCF model is trained automatically on startup and whenever you POST to `/train`.
- Training progress and loss will be shown in the terminal.
- The system is robust to missing columns (e.g., `timestamp`).

---

## Notes

- For best results, use real user interaction data with ratings.
- The project is for educational/demo purposes. For production, use a WSGI server and secure your endpoints.
- You can extend the system with more advanced models or data as needed.

---

## License

MIT (or your chosen license)

---

## Contact

For questions or contributions, open an issue or pull request on GitHub. 