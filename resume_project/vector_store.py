import chromadb

def store_in_chromadb(chunks, db_path="chromadb_data", collection_name="resume_collection"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    for i, chunk in enumerate(chunks):
        if chunk.embedding is not None:  # Skip chunks with failed embeddings
            collection.add(
                ids=[f"chunk_{i}"],
                documents=[chunk.text],
                embeddings=[chunk.embedding],
                metadatas=[chunk.metadata]
            )

    print(f"Stored {collection.count()} chunks into ChromaDB.")
    return collection