import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import pipeline

# Load longitudinal data (e.g., cognitive tests over time)
def load_longitudinal_data(file_path):
    # Load only necessary columns to save memory
    
    cols_to_load = ['Disease', 'Age', 'EDUC', 'MMSE', 'CDR']  # Adjust as needed
    data = pd.read_csv(file_path, usecols=cols_to_load)
    return data

# Preprocess Data
def preprocess_data(data):
    # Convert categorical data if any (for simplicity, assuming Disease is categorical)
    data['Disease'] = data['Disease'].astype('category').cat.codes

    # Split features and target
    features = data.drop(columns=['Disease'])  # Drop target column
    target = data['Disease']

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Reshape for LSTM (samples, timesteps, features)
    X = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
    y = target.values

    return X, y

# Build LSTM Model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        layers.LSTM(32, return_sequences=True, input_shape=input_shape),  # Reduced units
        layers.LSTM(16),  # Reduced units
        layers.Dense(1, activation='sigmoid')  # Binary classification for the target
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Analyze results with LLM
# Analyze results with LLM
def analyze_results_with_llm(summary_text):
    # Initialize a lightweight summarization pipeline from Hugging Face
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1, clean_up_tokenization_spaces=False)


    # Generate summary using the LLM
    summary = summarizer(summary_text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Main function
def main():
    file_path = 'longitudinal_data_sample_3590.csv'  # Replace with your CSV file path

    # Load and preprocess the data
    print("Loading longitudinal data...")
    data = load_longitudinal_data(file_path)
    X, y = preprocess_data(data)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    print("Training LSTM model...")
    history = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_val, y_val))  # Reduced epochs and batch size

    # Save the model
    model.save('./models/lstm_model.keras')

    print("Model saved to './models/lstm_model.h5'")

    # Prepare summary text based on some analysis (this should be customized)
    summary_text = "The dataset contains cognitive test scores for various diseases, with variables like Age, EDUC, MMSE, and CDR."
    print("Generating summary using LLM...")
    summary = analyze_results_with_llm(summary_text)
    print("LLM Summary:", summary)

if __name__ == "__main__":
    main()
