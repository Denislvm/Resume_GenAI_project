from data_loader import load_data
from chunker import create_chunks
from embedder import generate_embeddings, get_embed_model
from vector_store import store_in_chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def build_index(file_path="Resume.csv", limit=30):
    try:
        # Step 1: Load resumes
        documents = load_data(file_path, limit=limit)
        print(f"Loaded {len(documents)} documents")

        # Step 2: Chunk
        chunks = create_chunks(documents)
        print(f"Total chunks created: {len(chunks)}")

        # Step 3: Embeddings
        chunks = generate_embeddings(chunks)
        if not any(chunk.embedding for chunk in chunks):
            raise ValueError("No valid embeddings generated")
        print("First embeddings vector: ", chunks[0].embedding[:10] if chunks[0].embedding else "None")

        # Step 4: Store in ChromaDB
        collection = store_in_chromadb(chunks)

        # Step 5: Wrap with LlamaIndex
        embed_model = get_embed_model()
        vector_store = ChromaVectorStore(chroma_collection=collection)

        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        return index
    except Exception as e:
        print(f"Error in build_index: {str(e)}")
        raise

if __name__ == "__main__":
    index = build_index()
    print("Index built successfully")