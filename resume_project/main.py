# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Document
# from llama_index.core.node_parser import SentenceSplitter
# from sentence_transformers import SentenceTransformer
# import pandas as pd 

# df = pd.read_csv("Resume.csv", encoding="utf-8")

# df_subset = df.head(30)

# # Convert each row into document object
# documents = []
# for _, row in df_subset.iterrows():
#     resume_text = f"""
#     Resume_str: {row.get('Resume_str', '')}
#     Resume_html: {row.get('Resume_html', '')}
#     Category: {row.get('Category', '')}
#     """
#     documents.append(Document(text=resume_text.strip()))

# keywords = ["HR", "Administrator", "Specialist", "Director"]
# filtered_docs = [
#     doc for doc in documents
#     if any(keyword.lower() in doc.text.lower() for keyword in keywords)
# ]

# # Split into chunks 
# splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
# chunks = splitter.get_nodes_from_documents(filtered_docs)

# print(f"filtered documets: ", {len(filtered_docs)})
# print(f"Total chinks created: ", {len(chunks)})

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# for chunk in chunks:
#     chunk.embedding = embed_model.get_text_embedding(chunk.text)

# print("First embeddings vector: ", chunks[0].embedding[:10])

# if __name__=="__main__":
#     print("Good work because it works :D")


from data_loader import load_data
from chunker import create_chunks
from embedder import generate_embeddings
from vector_store import store_in_chromadb
from retriever import retrieve_candidates
from summarizer import generate_summary

if __name__ == "__main__":
    # Step 1: Load and filter documents
    documents = load_data("Resume.csv", limit=30)
    print(f"Filtered documents: {len(documents)}")

    # Step 2: Split into chunks
    chunks = create_chunks(documents)
    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Generate embeddings
    chunks = generate_embeddings(chunks)
    print("First embeddings vector: ", chunks[0].embedding[:10])

    # Step 4: Store embeddings in ChromaDB
    collection = store_in_chromadb(chunks)
    print("Project completed up to Step 4")

    # Step 5: retrieve candidates details
    candidates = retrieve_candidates("HR specialist with 5 years experience", n_results=2)
    for c in candidates:
        print("\nCandidate:")
        print("ID:", c["id"])
        print("Profession:", c["profession"])
        print("Experience:", c["experience"])
        print("Resume preview:", c["resume_preview"])

    # Step 6: Generate experience summary 
    summaries = generate_summary("HR specialist with 5 years experience")
    print("\nCandidate Summaries:")
    for s in summaries:
        print("-", s)  
