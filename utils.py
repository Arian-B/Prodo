import joblib
import os

def save_model(model, filepath):
    """
    Save a model to a file using joblib.
    
    Args:
        model: The model object to save.
        filepath (str): Path where the model will be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(model, filepath)
        print(f"Model successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filepath):
    """
    Load a model from a file using joblib.
    
    Args:
        filepath (str): Path to the saved model file.
    
    Returns:
        model: The loaded model object.
    """
    try:
        model = joblib.load(filepath)
        print(f"Model successfully loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None