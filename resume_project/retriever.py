import re
import chromadb

def extract_years_experience(text: str) -> str:
    """Try to extract years of experience from resume text."""
    match = re.search(r"(\d+)\s+years", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} years"
    return "Not specified"

def retrieve_candidates(query: str, n_results: int = 3, db_path="chromadb_data", collection_name="resume_collection"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    candidates = []
    for idx, doc in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][idx]
        candidate = {
            "id": results['ids'][0][idx],
            "profession": "Unknown",  # we don’t have a direct profession field in stored metadata
            "experience": extract_years_experience(doc),
            "resume_preview": doc[:200] + "..."
        }
        candidates.append(candidate)

    return candidates
