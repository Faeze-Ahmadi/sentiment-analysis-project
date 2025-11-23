import pickle

def load_objects(model_path="../model.pkl", vec_path="../vectorizer.pkl"):
    """Load model and vectorizer from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_sentiment(text, model, vectorizer):
    """Predict sentiment of a single text string."""

    # Transform text into TF-IDF vector
    X = vectorizer.transform([text])

    # Predict (0 = negative, 1 = positive)
    pred = model.predict(X)[0]

    return "POSITIVE 😊" if pred == 1 else "NEGATIVE 😡"


if __name__ == "__main__":
    model, vectorizer = load_objects()

    print("Sentiment Prediction Demo")
    print("-------------------------")

    while True:
        user_text = input("\nEnter a review (or type 'exit' to quit): ")

        if user_text.lower() == "exit":
            break

        result = predict_sentiment(user_text, model, vectorizer)
        print("Prediction:", result)
