import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

model = pickle.load(open("model/model.pkl", "rb"))
tf_idf = pickle.load(open("vectorizer/vectorizer.pkl", "rb"))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)


    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)


    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)


st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the msg/Email")

if st.button("predict"):

    # Step 1- Preprocess
    transformed_sms = transform_text(input_sms)

    # step 2- vetorize
    vector = tf_idf.transform([transformed_sms])

    # step 3- Predict
    result = model.predict(vector)[0]

    # step 4- display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")