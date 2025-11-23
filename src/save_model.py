import pickle
from load_data import load_reviews
from vectorize import vectorize_texts
from train_model import train_logreg


def save_objects(model, vectorizer, model_path="../model.pkl", vec_path="../vectorizer.pkl"):
    """Saves trained model and vectorizer to disk using pickle."""
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved successfully!")


if __name__ == "__main__":
    # 1) Load the data
    base = "../data"
    texts, labels = load_reviews(base)

    # 2) Vectorize
    X_train, X_test, y_train, y_test, vectorizer = vectorize_texts(texts, labels)

    # 3) Train the model
    model = train_logreg(X_train, y_train)

    # 4) Save everything
    save_objects(model, vectorizer)
