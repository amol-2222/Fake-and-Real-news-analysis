import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Load pre-trained Naive Bayes model and TF-IDF vectorizer
nb_model = joblib.load('final_nb_nlp_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# In[150]:
st.markdown('''
<style>
.stApp {
    
    background-color:#8DC8ED;
    align:center;\
    display:fill;\
    border-radius: false;\
    border-style: solid;\
    border-color:#000000;\
    border-style: false;\
    border-width: 2px;\
    color:Black;\
    font-size:15px;\
    font-family: Source Sans Pro;\
    background-color:#8DC8ED;\
    text-align:center;\
    letter-spacing:0.1px;\
    padding: 0.1em;">\
}
.sidebar {
    background-color: black;
}

.st-b7 {
    color: #8DC8ED;
}
.css-nlntq9 {
    font-family: Source Sans Pro;
}
</style>
''', unsafe_allow_html=True)



st.title("Fake and Real News Classification")

# Increase the size of the text input box by setting the height parameter
text_input = st.text_area("Enter a news article:", "Type here...", height=200)

prediction_button = st.button("Predict")
try:
    user_input_tfidf = tfidf_vectorizer.transform([text_input])
except Exception as e:
    st.error(f"Error during TF-IDF transformation: {e}")

if prediction_button:
    # Check if user input is not blank
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        # Preprocess user input and transform using TF-IDF
        user_input_tfidf = tfidf_vectorizer.transform([text_input])

        # Gather additional features
        title_length = len(text_input)  # Title length
        link_count = text_input.count('http')  # Count of links
        reuter_word = text_input.lower().count('reuter')  # Count of 'reuter' word
        text_blank = text_input.count(' ')  # Count of blank spaces
        
        # Combine TF-IDF features and additional features
        additional_features = np.array([[title_length, link_count, reuter_word, text_blank]])
        combined_features = np.hstack((user_input_tfidf.toarray(), additional_features))

        # Make a prediction using the loaded Naive Bayes model
        prediction = nb_model.predict(combined_features)
        probabilities = nb_model.predict_proba(combined_features)[0]

        # Display the prediction result and class probabilities
        if prediction[0] == 0:
            st.error("Fake News")
        else:
            st.success("Real News")
        Fake_prob= probabilities[0]
    True_prob= probabilities[1]
    if prediction == 0:
        if Fake_prob <= 0.2:
            st.progress(Fake_prob, text='Fakeness Chance')
        elif 0.4 >= Fake_prob > 0.2:
            st.progress(Fake_prob, text='Fakeness Chance')
        elif 0.6 >= Fake_prob > 0.4:
            st.progress(Fake_prob, text='Fakeness Chance')
        elif 0.8 >= Fake_prob > 0.6:
            st.progress(Fake_prob, text='Fakeness Chance')
        else:
            st.progress(Fake_prob, text='Fakeness Chance')
    elif prediction == 1:
        if True_prob <= 0.2:
            st.progress(True_prob, text='Truth Chance')
        elif 0.4 >= True_prob > 0.2:
            st.progress(True_prob, text='Truth Chance')
        elif 0.6 >= True_prob > 0.4:
            st.progress(True_prob, text='Truth Chance')
        elif 0.8 >= True_prob > 0.6:
            st.progress(True_prob, text='Truth Chance')
        else:
            st.progress(True_prob, text='Truth Chance')
