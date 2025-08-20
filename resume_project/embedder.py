from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def generate_embeddings(chunks, model_name="BAAI/bge-small-en-v1.5"):
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    for chunk in chunks:
        chunk.embedding = embed_model.get_text_embedding(chunk.text)
    return chunks
