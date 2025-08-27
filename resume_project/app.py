import streamlit as st
from main import build_index
from embedder import get_embed_model
from retriever import retrieve_candidates
from summarizer import generate_summary
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

# Configure LlamaIndex settings
Settings.embed_model = get_embed_model()

# Set local LLM
Settings.llm = HuggingFaceLLM(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tokenizer_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7},
    device_map="auto",
    model_kwargs={"load_in_4bit": True}  # 4-bit quantization for memory efficiency
)

# Build the index once
@st.cache_resource
def get_index(_retry=0):
    try:
        return build_index("Resume.csv", limit=30)
    except Exception as e:
        if _retry < 1:
            st.cache_resource.clear()
            return get_index(_retry + 1)
        st.error(f"Error building index: {str(e)}")
        return None

index = get_index()

if index is None:
    st.stop()

st.title("📂 Candidate Explorer")

# Search input
query = st.text_input("Search candidates (e.g., 'HR specialist with 5 years experience')", "")

with st.spinner("Retrieving candidates..."):
    try:
        candidates = retrieve_candidates(index, query, n_results=5 if query else 30)
    except Exception as e:
        st.error(f"Error retrieving candidates: {str(e)}")
        candidates = []

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
        format_func=lambda x: f"Candidate {x.get('id', 'Unknown')} ({x.get('profession', 'Unknown')}, {x.get('experience', 'Unknown')})"
    )

    if selected_candidate:
        st.write("### Candidate Details")
        st.json(selected_candidate)

        # Generate summary
        with st.spinner("Generating summary..."):
            try:
                summary = generate_summary(index, selected_candidate["text"])
                st.write("### Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")