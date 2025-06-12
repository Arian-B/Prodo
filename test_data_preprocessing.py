from data_preprocessing import load_data, preprocess_data

# Path to the sample dataset
filepath = 'D:/Coding/AI_PRSys/data/user_interactions.csv'

# Load the data
data = load_data(filepath)

if data is not None:
    # Preprocess the data
    preprocessed_data = preprocess_data(data)
    
    # Display the preprocessed data
    print("\nPreprocessed Data:")
    print(preprocessed_data)