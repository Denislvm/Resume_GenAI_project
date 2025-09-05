import streamlit as st
import json
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Load vector index (optional, kept for potential future use)
chroma_client = chromadb.PersistentClient(path="./vector_store")
chroma_collection = chroma_client.get_collection("resumes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

# Load pre-extracted data and summaries
candidates = json.load(open("candidates.json"))
full_texts = json.load(open("full_texts.json"))
summaries = json.load(open("summaries.json"))

# Web app
st.title("Candidate Resume Viewer (5 Random Resumes)")

# Create a dropdown to select a candidate
candidate_options = [
    f"{candidate['name']} - {candidate['profession']} ({candidate['years']} years experience)"
    for candidate in candidates
]
selected_candidate = st.selectbox("Select a Candidate", candidate_options)

selected_candidate_data = next(
    (c for c in candidates if f"{c['name']} - {c['profession']} ({c['years']} years experience)" == selected_candidate),
    None
)

if selected_candidate_data:
    st.subheader(selected_candidate)
    
    st.write("**Name**: " + selected_candidate_data['name'])
    st.write("**Profession**: " + selected_candidate_data['profession'])
    st.write("**Years of Experience**: " + str(selected_candidate_data['years']))

    st.write("### Detailed Information (Full Resume)")
    st.text_area(
        "Resume Text",
        full_texts[selected_candidate_data['id']],
        height=300,
        key=selected_candidate_data['id']
    )

    st.write("### Experience Summary")
    st.write(summaries.get(selected_candidate_data['id'], "No summary available."))
else:
    st.write("Please select a candidate to view details.")
    