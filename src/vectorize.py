from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def vectorize_texts(texts, labels, test_size=0.2):
    """
    Converts raw text data into TF-IDF vectors.
    Splits into train and test sets.
    Returns: X_train, X_test, y_train, y_test, vectorizer
    """

    # 1) Converting Text to Numbers with TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=10000,    # 10k is the best amount to start with.
        stop_words='english',  # Remove useless words like the, a, an
        ngram_range=(1, 2)     # unigram + bigram → much better performance
    )

    # Constructing the TF-IDF matrix
    X = vectorizer.fit_transform(texts)

    # Splitting data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, vectorizer


# Quick test
if __name__ == "__main__":
    from load_data import load_reviews

    base = "../data"
    texts, labels = load_reviews(base)

    X_train, X_test, y_train, y_test, vec = vectorize_texts(texts, labels)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Feature count:", len(vec.get_feature_names_out()))
