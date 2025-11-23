### *A Machine Learning Project Using TF-IDF and Logistic Regression*

This project implements a simple but effective **sentiment analysis** system trained on movie reviews.
The goal is to build a lightweight, understandable, and reusable model that can classify text as **positive** or **negative** sentiment.

It is intentionally built with clarity in mind, making it easy to learn, modify, and extend.

---

## **About This Project**

This project performs **binary sentiment classification** on raw movie reviews.
The workflow consists of:

1. **Loading raw text files** from `pos/` and `neg/`.
2. **Converting text into numerical vectors** using TF-IDF.
3. **Training a Logistic Regression classifier** to separate positive and negative sentiment.
4. **Saving the trained model & vectorizer** using `pickle`.
5. **Predicting sentiment** for new user-input text without retraining.

The entire pipeline is modular and easy to reuse or extend.

---

## **Project Structure**

```
sentiment-analysis-project/
│
├── data/
│   ├── neg/                  # Negative movie reviews (25,000)
│   └── pos/                  # Positive movie reviews (25,000)
│
├── model.pkl                 # Saved Logistic Regression model
├── vectorizer.pkl            # Saved TF-IDF vectorizer
│
├── src/
│   ├── load_data.py          # Loads and labels the dataset
│   ├── vectorize.py          # TF-IDF vectorization + train/test split
│   ├── train_model.py        # Trains and evaluates the model
│   ├── save_model.py         # Saves model.pkl and vectorizer.pkl
│   └── test_model.py         # Interactive prediction script
│
└── README.md
```

---

## **Training the Model (`train_model.py`)**

The training script performs the following steps:

* Loads positive and negative reviews.
* Assigns labels:

  * `1` → positive
  * `0` → negative
* Converts text into TF-IDF features (unigrams + bigrams).
* Splits the dataset into **80% train / 20% test**.
* Trains a Logistic Regression classifier.
* Prints accuracy + classification report.

Typical performance:

```
Accuracy: ~0.8884 (≈ 89%)
```

This is excellent for a traditional ML model.

---

## **Saving the Model (`save_model.py`)**

This script:

* Retrains the model (or loads it from your code)
* Saves:

  * `model.pkl`
  * `vectorizer.pkl`

Both files are stored in the project root.

These can be reused later for predictions without needing retraining.

---

## **Making Predictions (`test_model.py`)**

This script:

* Loads `model.pkl` and `vectorizer.pkl`
* Accepts user text input
* Vectorizes the input using the saved TF-IDF vectorizer
* Predicts sentiment using the trained model
* Returns:

```
POSITIVE
or
NEGATIVE
```

### Example:

```
Enter a review:
this movie was absolutely amazing

Prediction: POSITIVE
```

---

## **Key Concepts**

### **TF-IDF Vectorization**

TF-IDF transforms text into numerical vectors based on:

* Term importance within the document
* Rarity/uniqueness in the corpus

This helps highlight meaningful words while downplaying common ones.

### **Logistic Regression**

A simple but powerful algorithm for **binary classification**, ideal for:

* High-dimensional sparse TF-IDF vectors
* Fast and interpretable models
* Baseline NLP tasks

### **Saving & Loading with Pickle**

Using `pickle` allows:

* Saving trained models
* Loading later without retraining
* Fast inference (good for real applications)

---

## **How to Run the Project**

### 1. Clone the repository

```bash
git clone https://github.com/Faeze-Ahmadi/sentiment-analysis-project
cd sentiment-analysis-project
```

### 2. Install dependencies

```bash
pip install scikit-learn
```

(You can also add a `requirements.txt` later.)

### 3. Train the model

```bash
python src/train_model.py
```

### 4. Save the model for future use

```bash
python src/save_model.py
```

This produces:

```
model.pkl
vectorizer.pkl
```

### 5. Run the interactive prediction tool

```bash
python src/test_model.py
```

Then type your sentence.

---

## Additional Notes

* English **stop words** are removed to eliminate unhelpful common words.
* TF-IDF uses both **unigrams** and **bigrams** for richer features.
* Dataset split uses a fixed `random_state=42` for reproducibility.
* The project is intentionally minimalistic to focus on the ML core.

---

## Future Improvements

Possible extensions:

* Replace Logistic Regression with **SVM**, **Random Forest**, or **BERT**.
* Build a simple **web UI** using Streamlit or FastAPI.
* Perform **hyperparameter tuning** using GridSearchCV.
* Add data visualization (word clouds, frequency analysis).

---

## License

This project is open-source and available under the **MIT License**. Feel free to use, modify, and distribute.

---

## Contribution

Contributions are welcome! If you want to add new features (such as a web interface, expanded dataset, or more advanced models), please open an issue or submit a pull request.

