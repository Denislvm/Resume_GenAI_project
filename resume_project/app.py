import streamlit as st
from main import build_index
from embedder import get_embed_model
from retriever import retrieve_candidates
from summarizer import generate_summary
from llama_index.core import Settings
import json

# Configure LlamaIndex settings
Settings.embed_model = get_embed_model()
Settings.llm = None

# Build the index once
@st.cache_resource
def get_index():
    return build_index("Resume.csv", limit=30)

index = get_index()

st.title("📂 Candidate Explorer")

# Search input
query = st.text_input("Search candidates (e.g., 'HR specialist with 5 years experience')", "")

if query:
    with st.spinner("Retrieving candidates..."):
        candidates = retrieve_candidates(index, query, n_results=5)

    if not candidates:
        st.warning("No candidates found.")
    else:
        st.subheader("Candidates Found")

        # Ensure candidates is a list
        candidate_list = candidates if isinstance(candidates, list) else [candidates]

        # Candidate selection
        selected_candidate = st.selectbox(
            "Select a candidate:",
            options=candidate_list,
            format_func=lambda x: x.get("name", "Unknown") + f" ({x.get('profession', 'Unknown')})"
        )

        if selected_candidate:
            st.write("### Candidate Details")
            st.json(selected_candidate)

            # Generate summary for the selected candidate
            with st.spinner("Generating summary..."):
                summary = generate_summary(index, selected_candidate["text"], n_results=1)
            
            st.write("### Summary")
            st.write(summary)