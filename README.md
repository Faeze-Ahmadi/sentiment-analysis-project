### A Machine Learning Project Using TF-IDF and Logistic Regression

This project implements a simple but effective **sentiment analysis** system trained on movie reviews. The goal is to build a lightweight, understandable, and reusable model that can classify user-input texts as **positive** or **negative** sentiment.

---

## About This Project

In this project, raw text data is analyzed to determine the sentiment behind each review. The core idea consists of two main steps:

1. **Preprocessing and converting text into numerical vectors using TF-IDF**
2. **Training a Logistic Regression model to classify sentiment**

The entire pipeline is designed so you can easily retrain the model with new data or use the saved model to make predictions on new text inputs without retraining.

---

## How It Works

Since machines cannot understand raw text, we first convert the text into a numeric format they can process. Here, we use **TF-IDF (Term Frequency-Inverse Document Frequency)**, which represents texts as weighted numerical vectors reflecting the importance of words.

Once texts are vectorized, a **Logistic Regression classifier** learns patterns associated with positive and negative sentiments. After training, both the model and the TF-IDF vectorizer are saved. This enables quick sentiment predictions later without needing to train the model again.

---

## Project Structure

```
sentiment-analysis-project/
│
├── data/
│   ├── neg/                 
│   └── pos/                 
│
├── model/
│   ├── model.pkl            
│   └── vectorizer.pkl       
│
├── src/
│   ├── load_data.py        
│   ├── save_model.py        
│   ├── test_model.py        
│   ├── train_model.py       
│   └── vectorize.py        
│
├── README.md               

```

---

## Training the Model (`train.py`)

The training script performs the following steps:

* Loads the dataset containing positive and negative reviews.
* Assigns labels: 1 for positive reviews, 0 for negative reviews.
* Converts the text data into TF-IDF feature vectors.
* Splits data into training and testing sets.
* Trains a Logistic Regression model on the training data.
* Saves the trained model and vectorizer to disk for later use.

---

## Making Predictions (`predict.py`)

The prediction script:

* Loads the saved model and vectorizer from disk.
* Accepts text input from the user.
* Converts the input text to a TF-IDF vector using the saved vectorizer.
* Uses the trained model to predict whether the sentiment is positive or negative.
* Outputs the prediction in a user-friendly format.

---

## Key Concepts

* **TF-IDF Vectorization**: Transforms text into numerical features that reflect the importance of each word relative to the document and the entire dataset.
* **Logistic Regression**: A simple yet powerful machine learning algorithm for binary classification tasks like sentiment analysis.
* **Saving and Loading Models**: Using `pickle` to serialize the trained model and vectorizer, enabling reuse without retraining.

---

## How to Run the Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Faeze-Ahmadi/sentiment-analysis-project
   cd sentiment-analysis
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**

   ```bash
   python src/train.py
   ```

   This will generate `model.pkl` and `vectorizer.pkl` files inside the model directory.

4. **Run the prediction script:**

   ```bash
   python src/predict.py
   ```

   Enter your review text when prompted, and see the sentiment prediction.

---

## Additional Notes

* The project uses **English stop words** to improve vectorization quality by ignoring common but unimportant words like "the", "a", "and".
* The TF-IDF vectorizer considers both single words (unigrams) and two-word sequences (bigrams) for better performance.
* The training splits the data into 80% training and 20% testing with a fixed random seed for reproducibility.

---

## License

This project is open-source and available under the **MIT License**. Feel free to use, modify, and distribute.

---

## Contribution

Contributions are welcome! If you want to add new features (such as a web interface, expanded dataset, or more advanced models), please open an issue or submit a pull request.

