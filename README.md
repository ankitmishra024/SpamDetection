# Spam SMS Detection Project

## Overview

This project is a **Spam SMS Detection** system that classifies messages as spam or ham (not spam) using **Natural Language Processing (NLP)** techniques and **Machine Learning models**. The dataset used for training consists of labeled SMS messages.

## Features

- **Data Preprocessing**: Cleaning and transforming text data (lowercasing, tokenization, stopword removal, stemming).
- **Exploratory Data Analysis (EDA)**: Visualizing and analyzing message statistics.
- **Feature Extraction**: Using **TF-IDF** and **CountVectorizer**.
- **Model Training & Evaluation**: Naïve Bayes classifiers (GaussianNB, MultinomialNB, BernoulliNB) were trained and tested.
- **Deployment**: A **Streamlit** web application to run the model on a local host.

## Installation

### 1. Clone the Repository

```sh
    git clone https://github.com/ankitmishra024/SpamDetection.git
    cd SpamDetection
```

### 2. Install Dependencies

Ensure you have **Python 3.8+** installed. Then, install the required libraries:

```sh
    pip install -r requirements.txt
```

### 3. Run the Streamlit App

```sh
    streamlit run app.py
```

This will launch the app in your browser at **localhost:8501**.

## Usage

1. **Upload** or enter an SMS message in the text input.
2. Click **Predict** to classify the message as spam or ham.
3. View the prediction results instantly.

## Model & Performance

- **Best Model**: **Multinomial Naïve Bayes (MNB) with TF-IDF vectorization**.
- **Accuracy**: \~97%
- **Precision**: \~99%

## Files & Directory Structure

```
spam-sms-detection/
│── model/
│   ├── model.pkl          # Saved trained model
│── vectorizer/
│   ├── vectorizer.pkl     # TF-IDF vectorizer
│── app.py                 # Streamlit web app
│── data/
│   │──spam.csv            # Dataset
│── main.py                # Main script for training & evaluation
│── requirements.txt       # Required Python libraries
│── README.md              # Project documentation
```

## Dependencies

- **Python 3.8+**
- **Pandas, NumPy** (Data handling)
- **NLTK** (Text preprocessing)
- **Scikit-learn** (Machine Learning models)
- **Matplotlib, Seaborn** (Data visualization)
- **Streamlit** (Web app for deployment on local)
