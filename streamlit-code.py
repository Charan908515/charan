import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the trained model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("next-word-prediction.h5")

model = load_model()

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(text, model, tokenizer, max_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

st.title("Next Word Prediction")
user_input = st.text_input("Enter a sentence:")

if st.button("Predict Next Word"):
    if user_input:
        next_word = predict_next_word(user_input, model, tokenizer, max_len=20)  # Adjust max_len accordingly
        st.write(f"Predicted next word: **{next_word}**")
    else:
        st.write("Please enter some text.")
