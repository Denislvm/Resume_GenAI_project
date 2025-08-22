from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_embed_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Return a HuggingFace embedding model instance"""
    return HuggingFaceEmbedding(model_name=model_name)

def generate_embeddings(chunks, model_name: str = "BAAI/bge-small-en-v1.5"):
    """Generate embeddings for a list of text chunks"""
    embed_model = get_embed_model(model_name)
    for chunk in chunks:
        chunk.embedding = embed_model.get_text_embedding(chunk.text)
    return chunks
