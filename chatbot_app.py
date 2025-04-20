import nltk
import string
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Load and Read the Text File ===
file_path = "MUFATE_SACCO_CLEANED_FOR_CHATBOT.txt"

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
except FileNotFoundError:
    st.error(f"File not found: {file_path}")
    st.stop()

# === Step 1: Split raw text into chunks ===
chunks = raw_text.split("\n\n")

# === Step 2: Clean chunks — combine short headings with the following paragraph ===
cleaned_chunks = []
i = 0
while i < len(chunks):
    current = chunks[i].strip()
    if len(current) < 40 and (i + 1) < len(chunks):
        # Combine short heading with the next chunk
        combined = current + " " + chunks[i + 1].strip()
        cleaned_chunks.append(combined)
        i += 2
    else:
        cleaned_chunks.append(current)
        i += 1

chunks = cleaned_chunks

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

# === Preprocess Each Chunk for Comparison ===
corpus = [preprocess(chunk) for chunk in chunks]

# === Matching Logic: Find the Most Relevant Chunk ===
# Create TF-IDF vectorizer from original cleaned chunks
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(chunks)

# === New Function to Match User Query Using Cosine Similarity ===
def get_most_relevant_chunk(query):
    query_tfidf = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

    # Boost chunks that contain the exact query phrase (case-insensitive)
    for i, chunk in enumerate(chunks):
        if query.lower().strip(':') in chunk.lower():
            similarity_scores[i] += 0.5  # Boost match score

    # Pick the highest scoring chunk
    top_index = similarity_scores.argmax()
    top_score = similarity_scores[top_index]

    if top_score > 0.1:
        return chunks[top_index]
    else:
        return "Sorry, I couldn't find anything relevant. Try rephrasing your question."

# === Chatbot Logic ===
def chatbot(question):
    return get_most_relevant_chunk(question)

# === Streamlit Web UI ===
def main():
    st.set_page_config(page_title="Mudete SACCO Chatbot")
    st.title("🤝 Mudete SACCO Chatbot")
    st.markdown("Ask me anything about **Mudete SACCO** – loans, savings, membership, mobile banking, etc.")

    user_input = st.text_input("You:")

    if st.button("Submit") and user_input:
        response = chatbot(user_input)
        st.markdown(f"**Chatbot:** {response}")

# === Run the App ===
if __name__ == "__main__":
    main()
