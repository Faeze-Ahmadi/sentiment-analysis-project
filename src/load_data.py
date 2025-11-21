import os

def load_reviews(base_path):
    """
    Reads all .txt files from pos and neg folders and returns:
    - texts: list of string (each review)
    - labels: list of int (1 for positive, 0 for negative)
    """

    texts = []
    labels = []

    # Paths to pos and neg folders
    pos_path = os.path.join(base_path, "pos")
    neg_path = os.path.join(base_path, "neg")

    # ---- Read POSITIVE reviews (label = 1) ----
    for filename in os.listdir(pos_path):
        if filename.endswith(".txt"):
            with open(os.path.join(pos_path, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(1)

    # ---- Read NEGATIVE reviews (label = 0) ----
    for filename in os.listdir(neg_path):
        if filename.endswith(".txt"):
            with open(os.path.join(neg_path, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(0)

    return texts, labels


# Quick test
if __name__ == "__main__":
    base = "../data"   # because src is inside project root
    texts, labels = load_reviews(base)
    print("Number of texts:", len(texts))
    print("Number of labels:", len(labels))
    print("Example text:", texts[0][:200])
