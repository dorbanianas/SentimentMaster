import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from src.preprocessing import preprocess_text
from src.model import create_model
from tqdm import tqdm
import os
import nltk

def train_model():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('data/IMDB Dataset.csv')
    print("Dataset loaded.")

    # Apply preprocess_text function with tqdm
    tqdm.pandas()

    # Preprocess text
    
    # Check if the sentiment classifier model exists
    if not os.path.exists('data/imdb_prerocessed_dataset.csv'):
        print("Preprocessing text...")
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        df['review'] = df['review'].progress_apply(preprocess_text)
        print("Text preprocessing complete.")
        print("Saving preprocessed data...")
        df.to_csv('data/imdb_prerocessed_dataset.csv', index=False)
        print("Preprocessed data saved.")
    else:
        print("Preprocessed data found.")
        print("Loading preprocessed data...")
        df = pd.read_csv('data/imdb_prerocessed_dataset.csv')
        print("Preprocessed data loaded.")

    # Split dataset into training and testing sets
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
    print("Dataset split complete.")

    # Create and train the model
    model = create_model()
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("Evaluating model...")
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the model
    print("Saving model...")
    joblib.dump(model, 'models/sentiment_classifier.pkl')
    print("Model saved.")