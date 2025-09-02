from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
import json
import re
from collections import defaultdict

def retrieve_candidates(index: VectorStoreIndex, query: str, n_results: int = 5):
    """Retrieve top-N candidates based on a query."""
    if not query.strip():
        top_k = 1000
        retrieval_query = "candidate resume"
    else:
        top_k = n_results * 3
        retrieval_query = query

    retriever = index.as_retriever(similarity_top_k=top_k)
    query_engine = RetrieverQueryEngine.from_args(retriever)

    if "hr specialist" in query.lower():
        filter_query = """
        Only include candidates whose Category is 'HR' or whose resume contains 'HR Specialist'.
        """
    else:
        filter_query = ""

    extraction_prompt = f"""
    {filter_query}
    Extract:
    1. Candidate ID (from 'Candidate ID:' in the text)
    2. Profession / Job Title (from Category or explicit mentions)
    3. Years of commercial experience (calculate from dates or mentions; if unclear, return "Unknown")
    Return in JSON format with keys: id, profession, experience, text (full resume text).
    """

    response = query_engine.query(extraction_prompt + f"\n\nQuery: {retrieval_query}")
    
    try:
        cleaned_response = re.sub(r'^Context information is below\..*?\n\n', '', response.response, flags=re.DOTALL)
        candidates = json.loads(cleaned_response)
        if not isinstance(candidates, list):
            candidates = [candidates]
    except (json.JSONDecodeError, ValueError):
        candidate_data = defaultdict(lambda: {"texts": [], "metadata": None})
        for node in response.source_nodes:
            cid = node.metadata.get("candidate_id")
            if cid:
                candidate_data[cid]["texts"].append(node.text)
                candidate_data[cid]["metadata"] = node.metadata

        candidates = []
        for cid, data in candidate_data.items():
            full_text = "\n".join(data["texts"])
            profession = data["metadata"].get("category", "Unknown")
            if "hr specialist" in query.lower() and profession != "HR" and "HR Specialist" not in full_text:
                continue
            experience_match = re.search(r'(\d+\+?\s*years?\s*(?:of\s*)?(?:experience|in))', full_text, re.IGNORECASE)
            experience = experience_match.group(1) if experience_match else "Unknown"
            candidates.append({
                "id": cid,
                "profession": profession,
                "experience": experience,
                "text": full_text
            })

    return candidates[:n_results] if query.strip() else candidates