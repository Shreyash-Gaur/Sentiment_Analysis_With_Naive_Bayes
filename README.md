# Sentiment Analysis Project

## Project Overview

This project demonstrates a Sentiment Analysis model using the Naive Bayes algorithm. It aims to classify the sentiment of tweets as either positive or negative based on their textual content. The project utilizes the NLTK library for natural language processing and preprocessing tasks. The implementation showcases various stages of the machine learning pipeline, including data preprocessing, feature extraction, model training, evaluation, and prediction.

## Key Highlights
- **Comprehensive Preprocessing**: The project includes thorough text preprocessing steps such as tokenization, stemming, and stop words removal to ensure high-quality input for the model.
- **Frequency-Based Feature Extraction**: Using a frequency dictionary, the model effectively captures word occurrences to distinguish between positive and negative sentiments.
- **Naive Bayes Classifier**: The Naive Bayes algorithm is implemented to leverage the probabilistic relationships between words and sentiments.
- **Custom Predictions**: The notebook allows users to input their own tweets and obtain sentiment predictions from the trained model.

## Files in the Repository

1. **`Sentiment_Analysis_(Naive Bayes).ipynb`**: Jupyter Notebook containing the implementation of the Naive Bayes sentiment analysis model.
2. **`utils.py`**: Python script with utility functions for preprocessing tweets, looking up word frequencies, and plotting confidence ellipses.

## Notebook Overview

### Requirements
- Python 3.x
- Jupyter Notebook
- Required libraries: numpy, pandas, nltk, sklearn

You can install the required libraries using:
```bash
pip install numpy pandas nltk scikit-learn os
```

### Imports

The following libraries and modules are imported:
```python
from utils import process_tweet, lookup
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd

nltk.download('twitter_samples')
nltk.download('stopwords')
```

### Contents of the Notebook
1. **Introduction**: Brief description of the project and the Naive Bayes algorithm.
2. **Importing Libraries**: Importing the necessary libraries for the project.
3. **Loading the Dataset**: Loading the dataset of tweets and preparing it for analysis.
4. **Text Preprocessing**: Cleaning and preprocessing the tweets, including tokenization, stemming, and stop words removal.
5. **Feature Extraction**: Extracting features from the preprocessed text for model training.
6. **Training Naive Bayes Model**: Training the Naive Bayes model using the processed dataset.
7. **Evaluating the Model**: Evaluating the performance of the model using test data.
8. **Error Analysis**: Analyzing the errors made by the model to understand its weaknesses.
9. **Predicting Custom Tweets**: Testing the model with custom tweets to predict their sentiment.

### Model Evaluation and Prediction

The notebook includes functions to predict sentiment and test the Naive Bayes model:
- `naive_bayes_predict(tweet, logprior, loglikelihood)`: Predicts the sentiment of a tweet.
- `test_naive_bayes(test_x, test_y, logprior, loglikelihood)`: Evaluates the model's accuracy on a test set.

Examples of predictions on sample tweets, error analysis, and testing with custom tweets are also provided.

### Key Functions

- **`process_tweet(tweet)`**: Cleans and preprocesses a tweet by removing unnecessary elements (stemming, removing stop words) and tokenizing the text.
- **`build_freqs(tweets, ys)`**: Builds a frequency dictionary for each word in the dataset.
- **`test_lookup(func)`**: Tests the `lookup` function for correctness.
- **`lookup(freqs, word, label)`**: Looks up the frequency of a given word and label pair in a dictionary.
- **`confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs)`**: Creates a plot of the covariance confidence ellipse of `x` and `y`.
- **`train_naive_bayes(freqs, train_x, train_y)`**: Trains the Naive Bayes model using the frequency dictionary.
- **`naive_bayes_predict(tweet, logprior, loglikelihood)`**: Predicts the sentiment of a given tweet using the trained model.

### Results

The notebook shows examples of processed tweets and trained model parameters, along with the accuracy of the Naive Bayes model on the test set.

### Custom Predictions
To predict the sentiment of your own tweet, modify the `my_tweet` variable in the notebook and run the corresponding cell:
```python
my_tweet = 'Your custom tweet here'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print(p)
```

## Usage
1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```
2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Sentiment_Analysis_(Naive_Bayes).ipynb
   ```
3. **Run the cells sequentially** to execute the code and observe the output. The notebook will guide you through the process of loading data, preprocessing tweets, training the Naive Bayes model, and evaluating its performance.

## Experience
Working on this project enhanced my understanding of natural language processing and machine learning techniques. I gained hands-on experience with text preprocessing, feature engineering, and the implementation of the Naive Bayes algorithm. Additionally, I improved my skills in Python programming and Jupyter Notebook usage.

Feel free to explore the project, run the notebook, and test the model with your own tweets. Your feedback and suggestions are welcome!

## Conclusion
In this project I provided a comprehensive guide for implementing a sentiment analysis model using Naive Bayes. It includes detailed explanations, code, and examples to help understand the process from data preprocessing to model evaluation and prediction.

Thank you for checking out my Sentiment Analysis project.
