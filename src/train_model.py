from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from load_data import load_reviews
from vectorize import vectorize_texts


def train_logreg(X_train, y_train):
    """
    Train a Logistic Regression classifier on TF-IDF features.
    Returns the trained model.
    """
    model = LogisticRegression(
        # We let it run for 1000 times until it converges well.
        max_iter=1000,
        n_jobs=-1           # Use all CPU cores for more speed
    )
    model.fit(X_train, y_train)
    # For my own test, the output was slow to print because the data was too heavy.
    print("Model training finished!")
    return model


if __name__ == "__main__":
    # 1) Load raw texts and labels
    base = "../data"
    texts, labels = load_reviews(base)

    # 2) Convert texts to TF-IDF vectors and split into train/test
    X_train, X_test, y_train, y_test, vectorizer = vectorize_texts(
        texts, labels)

    # 3) Train the Logistic Regression model
    model = train_logreg(X_train, y_train)

    # 4) Predict on the test set
    y_pred = model.predict(X_test)

    # 5) Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred,
          target_names=["negative", "positive"]))
