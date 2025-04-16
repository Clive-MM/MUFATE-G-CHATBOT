# chatbot_app.py

import nltk
import string
import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === Load and Read the Text File ===
file_path = "MUDETE FACTORY TEA GROWERS SAVINGS.txt"  # File must be in the same folder

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read().replace('\n', ' ')
except FileNotFoundError:
    st.error(f"❌ File not found: {file_path}")
    st.stop()

# === Tokenize the Text into Sentences ===
sentences = sent_tokenize(raw_text)

# === Preprocessing Function ===
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    cleaned_words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word.lower() not in stopwords.words('english') and word not in string.punctuation
    ]
    return cleaned_words

# === Preprocess Each Sentence in the Corpus ===
corpus = [preprocess(sentence) for sentence in sentences]

# === Find the Most Relevant Sentence Using Jaccard Similarity ===
def get_most_relevant_sentence(query):
    query_words = preprocess(query)
    max_similarity = 0
    best_match = ""

    for i, sentence_words in enumerate(corpus):
        similarity = len(set(query_words).intersection(sentence_words)) / float(len(set(query_words).union(sentence_words)))
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = sentences[i]

    return best_match if best_match else "Sorry, I couldn't find anything relevant. Try rephrasing your question."

# === Chatbot Logic ===
def chatbot(question):
    return get_most_relevant_sentence(question)

# === Streamlit Web Interface ===
def main():
    st.set_page_config(page_title="Mudete SACCO Chatbot")
    st.title("🤝 Mudete SACCO Chatbot")
    st.markdown("Ask me anything about **Mudete SACCO** – loans, savings, membership, etc.")

    user_input = st.text_input("You:")

    if st.button("Submit") and user_input:
        response = chatbot(user_input)
        st.markdown(f"**Chatbot:** {response}")

# === Run the Streamlit App ===
if __name__ == "__main__":
    main()
