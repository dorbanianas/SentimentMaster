import os
import joblib
import sys
from src import train, predict


def main():
    print(8*"=" + " Sentiment Analysis " + 8*"=")
    # Check if the sentiment classifier model exists
    if not os.path.exists('models/sentiment_classifier.pkl'):
        print("Sentiment classifier model not found. Training the model...")
        train.train_model()
        print("Model training complete.")
    else:
        print("Sentiment classifier model found.")

    # Load the model
    model = joblib.load('models/sentiment_classifier.pkl')

    # Option to make predictions
    while True:
        user_input = input("\nEnter a text to predict sentiment (type 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Exiting...")
            sys.exit()
        else:
            sentiment = predict.predict_sentiment(model, user_input)
            print("Predicted sentiment:", sentiment)

if __name__ == "__main__":
    main()
