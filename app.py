import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load Pre-trained Model
def load_model():
    model = tf.keras.models.load_model('modified_SDM10.h5')
    return model

# Preprocessing Function
def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Tokenizer
df = pd.read_csv('ML_HEADLINE_DATASET.csv')
df['headline'] = df['headline'].apply(preprocess_text)
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df['headline'])

# Load model
model = load_model()

# Preprocess and Predict Function
def preprocess_and_predict(sentence):
    preprocessed_sentence = preprocess_text(sentence)
    sequence = tokenizer.texts_to_sequences([preprocessed_sentence])
    if not sequence or not sequence[0]:
        return None
    
    # Pad the sequences
    X = pad_sequences(sequence, maxlen=150)
    
    # Predict sarcasm
    prediction = model.predict(X)
    return prediction[0][0] if prediction is not None and len(prediction) > 0 else None

# Feedback and Retrain Functions
def provide_feedback(sentence, is_sarcastic):
    with open('feedback.csv', 'a') as f:
        f.write(f"{sentence},{is_sarcastic}\n")
    st.success("Feedback has been recorded. Thank you!")

def retrain_model(sentence, is_sarcastic):
    # Convert the single sentence for retraining
    preprocessed_sentence = preprocess_text(sentence)
    sequence = tokenizer.texts_to_sequences([preprocessed_sentence])
    X_train = pad_sequences(sequence, maxlen=150)
    y_train = np.array([is_sarcastic])
    
    # Retrain the model with the new data
    model.fit(X_train, y_train, epochs=1, batch_size=1)
    model.save('modified_SDM10.h5')

# Feedback Loop Function
def feedback_loop():
    # Initialize session state variables if they don't exist
    if 'input_idx' not in st.session_state:
        st.session_state.input_idx = 0
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = False

    st.markdown("<p style='font-size:50px; color: pink; text-align:center; width:fit-content; margin:auto; margin-bottom:50px; font-weight:600;font-family: \"Lucida Console\", \"Courier New\", monospace;'>SARCASM DETECTION</p>", unsafe_allow_html=True)
    
    # Only show the input field if feedback hasn't been given yet
    if not st.session_state.feedback_given:
        user_input = st.text_input("Enter a headline (or 'quit' to exit):", key=f"input_{st.session_state.input_idx}")
        if user_input.lower() == "quit" or not user_input.strip():
            st.stop()
        sarcasm_prob = preprocess_and_predict(user_input)
        if sarcasm_prob is not None:
            predicted_class = 'sarcastic' if sarcasm_prob > 0.5 else 'not sarcastic'
            st.write(f"This headline is predicted as {predicted_class}.")
            feedback = st.radio("Was the prediction correct?", ("Yes", "No"), key=f"feedback_{st.session_state.input_idx}")
            if feedback.lower() == "no":
                corrected_label = 0 if predicted_class == 'sarcastic' else 1
                provide_feedback(user_input, corrected_label)
                retrain_model(user_input, corrected_label)
                st.success("The model has been updated with your feedback.")
                # Set the feedback_given state to True to prevent further input
                st.session_state.feedback_given = True
                # Increment the input index for the next input
                st.session_state.input_idx += 1

    # Button to allow the user to give another headline
    if st.button("Enter another headline"):
        # Reset the feedback_given state to False to show the input field again
        st.session_state.feedback_given = False
        # Clear the previous input and messages
        st.experimental_rerun()

# Run the feedback loop
feedback_loop()
