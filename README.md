# Tweet Sentiment Analysis with RNN

## Introduction

This project focuses on building a Recurrent Neural Network (RNN) for sentiment analysis of airline tweets. The goal is to classify tweets as positive, neutral, or negative based on their content.

## Dataset

The dataset used in this analysis is loaded from a CSV file named `Tweets.csv`. It contains various features related to airline tweets, including:

* `tweet_id`

* `airline_sentiment` (the target variable: positive, neutral, or negative)

* `airline_sentiment_confidence`

* `negativereason`

* `airline`

* `text` (the main content of the tweet)

* Other related metadata

For this analysis, only the `text` and `airline_sentiment` columns are used, and the dataset is limited to the first 14,000 rows.

## Project Goal

The primary goal of this project is to implement an RNN model capable of accurately classifying the sentiment of airline tweets.

## Key Steps

The notebook covers the following key steps:

* **Importing Libraries**: Essential libraries for data manipulation (`numpy`, `pandas`), and building neural networks (`torch`, `torch.nn`) are imported.

* **Data Loading and Initial Exploration**: The `Tweets.csv` dataset is loaded, and a preliminary inspection of the data, including the first few rows and the distribution of airline sentiment, is performed.

* **Data Preprocessing**:

  * Irrelevant columns are excluded, and only `text` and `airline_sentiment` are retained.

  * Text data is cleaned by removing punctuation, converting to lowercase, and splitting into individual words.

  * Web addresses, Twitter IDs (mentions), and digits are removed from the tweet text.

* **Encoding the Words**:

  * A vocabulary is built by mapping unique words to integers.

  * Each tweet is then converted into a sequence of these integer-encoded words.

* **Label Conversion**: The categorical sentiment labels (positive, neutral, negative) are converted into numerical representations (0 and 1, where positive and neutral are mapped to 1, and negative to 0).

* **Padding and Truncating Sequences**: A `pad_features` function is defined to ensure all tweet sequences have the same length (30 in this case) by padding shorter sequences with zeros and truncating longer ones.

* **Splitting Data**: The preprocessed data is split into training, validation, and test sets (80% for training, 10% for validation, and 10% for testing).

* **Creating DataLoaders**: `TensorDataset` and `DataLoader` from PyTorch are used to create iterable batches of data for efficient training and evaluation.

* **GPU Availability Check**: The notebook checks for GPU availability to leverage it for faster training if possible.

* **RNN Model Definition**: A `SentimentRNN` class is defined, which is a recurrent neural network model built using PyTorch's `nn.Module`. This model includes an embedding layer, an LSTM layer, dropout for regularization, and a linear output layer with a sigmoid activation function for binary classification.

## How to Run the Notebook

1. Ensure you have a Python environment with the necessary libraries installed. You can install them using pip:
 pip install numpy pandas torch

2. Make sure the `Tweets.csv` dataset is in the same directory as your notebook, or update the path in the code to point to the correct location of the file.

3. Open the `tweet.ipynb` file using Jupyter Notebook or JupyterLab.

4. Execute the cells sequentially to perform the data preprocessing, model definition, and subsequent training (training loop not fully shown in the provided snippet but would follow the model definition).

## Conclusion

This notebook demonstrates the foundational steps for performing sentiment analysis on text data using a Recurrent Neural Network (RNN) built with PyTorch, covering data preparation, encoding, and model architecture.
