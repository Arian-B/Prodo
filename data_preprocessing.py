import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """
    Load user interaction data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file containing user interactions.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data successfully loaded from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the data by handling missing values and normalizing features.
    
    Args:
        data (pd.DataFrame): Raw data loaded from the CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Handle missing values (fill with 0)
    data.fillna(0, inplace=True)
    print("Missing values handled.")
    
    # Normalize numerical features (e.g., clicks, ratings)
    scaler = MinMaxScaler()
    numerical_features = ['clicks', 'rating']
    if all(feature in data.columns for feature in numerical_features):
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        print("Numerical features normalized.")
    else:
        print("Some numerical features are missing in the dataset.")
    
    return data