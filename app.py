import string

import nltk
import streamlit as st
import pickle

from nltk import PorterStemmer
from nltk.corpus import stopwords

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message : ")


ps = PorterStemmer()


def transform_text(text):
    # lowercase
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # removing special characters
    y = []
    for i in text:
        if(i.isalnum() or i.isalpha()):
            y.append(i)

    text = y[:]
    y.clear()

    # removing stop words and punctuations
    # we did it outside function : nltk.download('stopwords')
    for i in text:
        if i not in stopwords.words('English') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # stemming
    for i in text:
        y.append(ps.stem(i))

    text = y[:]
    y.clear()
    return " ".join(text)


if st.button('Predict'):
    # now we need to do 4 steps :
    # 1.) preprocess

    transformed_sms = transform_text(input_sms)

    # 2.) vectorize

    vector_input = tfidf.transform([transformed_sms])

    # 3.) predict

    result = model.predict(vector_input)[0]

    # 4.) display

    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")