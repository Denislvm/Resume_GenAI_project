import chromadb
from transformers import pipeline

def generate_summary(query: str, db_path="chromadb_data", collection_name="resume_collection"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    summarizer = pipeline("summarization", model="google/flan-t5-small")

    summaries = []
    for doc in results["documents"][0]:
        text = doc[:500]
        summary = summarizer(text, max_length=100, min_length=20, do_sample=False)
        summaries.append(summary[0]["summary_text"])

    return summaries
