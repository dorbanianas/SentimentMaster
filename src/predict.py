import joblib

def predict_sentiment(model, text):
    # Make prediction
    sentiment = model.predict([text])[0]
    return sentiment
