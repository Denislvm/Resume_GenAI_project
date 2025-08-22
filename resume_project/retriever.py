from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
import json
import re

def retrieve_candidates(index: VectorStoreIndex, query: str, n_results: int = 5):
    """
    Retrieve top-N candidates based on a query (name, profession, years of experience).
    """
    retriever = index.as_retriever(similarity_top_k=n_results)
    query_engine = RetrieverQueryEngine.from_args(retriever)

    # Filter for HR Specialists if query contains "hr specialist"
    if "hr specialist" in query.lower():
        filter_query = """
        Only include candidates whose Category is 'HR' or whose resume contains 'HR Specialist' in the profession or job title.
        """
    else:
        filter_query = ""

    extraction_prompt = f"""
    {filter_query}
    Extract the following information from the candidate resume:
    1. Candidate full name (if available, else return "Unknown")
    2. Profession / Job Title
    3. Years of commercial experience (calculate based on experience dates or explicit mentions; if unclear, estimate or return "Unknown")
    Return results in JSON format with keys: name, profession, experience, text (the full resume text).
    """

    response = query_engine.query(extraction_prompt + f"\n\nQuery: {query}")
    
    # Attempt to parse the response as JSON
    try:
        # Clean the response to remove any "Context information" prefix
        cleaned_response = re.sub(r'^Context information is below\..*?\n\n', '', response.response, flags=re.DOTALL)
        candidates = json.loads(cleaned_response)
    except json.JSONDecodeError:
        # Fallback: Manually extract information from the response
        candidates = []
        for node in response.source_nodes:
            resume_text = node.node.text
            # Extract name (assuming not available in dataset)
            name = "Unknown"
            # Extract profession (from Category or resume text)
            profession = node.node.metadata.get("category", "Unknown")
            if "HR Specialist" in resume_text:
                profession = "HR Specialist"
            # Extract years of experience
            experience_match = re.search(r'(\d+\+?\s*years?\s*(?:of\s*)?(?:experience|in))', resume_text, re.IGNORECASE)
            experience = experience_match.group(1) if experience_match else "Unknown"
            candidates.append({
                "name": name,
                "profession": profession,
                "experience": experience,
                "text": resume_text
            })

    return candidates