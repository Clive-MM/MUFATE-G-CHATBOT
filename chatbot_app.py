import nltk
import string
import streamlit as st
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Load spaCy model for NER ===
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# === Load and Read the Text File ===
file_path = "MUFATE_SACCO_CLEANED_RESTRUCTURED.txt"

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
except FileNotFoundError:
    st.error(f"File not found: {file_path}")
    st.stop()

# Split on double newlines 
chunks = raw_text.strip().split("\n\n")

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

# === Match User Query Using Cosine Similarity ===
def get_most_relevant_chunk(query):
    query_tfidf = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

    for i, chunk in enumerate(chunks):
        title_line = chunk.split("\n")[0]
        if query.lower() in title_line.lower():
            similarity_scores[i] += 0.5
        if "contact us" in title_line.lower():
            if "contact" in query.lower() or "call" in query.lower() or "reach" in query.lower():
                return chunk  

    top_index = similarity_scores.argmax()
    if similarity_scores[top_index] > 0.1:
        return chunks[top_index]
    else:
        return "Sorry, I couldn't find anything relevant. Try rephrasing your question."

# === Chatbot Logic ===
def chatbot(question):
    return get_most_relevant_chunk(question)

# === Streamlit Web UI ===
def main():
    st.set_page_config(page_title="MUFATE G SACCO Chatbot")
    st.title(" MUFATE G SACCO Chatbot")
    st.markdown("Ask me anything about **MUFATE G SACCO** – loans, savings, membership, mobile banking, etc.")

    user_input = st.text_input("You:")

    if st.button("Submit") and user_input:
        entities = extract_entities(user_input)
        response = chatbot(user_input)
        st.markdown(f"**Chatbot:** {response}")

        if entities:
            st.markdown("\n**🔍 Detected Entities:**")
            for entity in entities:
                st.write(f"- {entity[0]} ({entity[1]})")

# === Run the App ===
if __name__ == "__main__":
    main()
