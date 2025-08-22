from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex

def generate_summary(index: VectorStoreIndex, resume_text: str, n_results: int = 1):
    """
    Generates a summary of strongest skills & professional highlights for a specific resume.
    """
    retriever = index.as_retriever(similarity_top_k=n_results)
    query_engine = RetrieverQueryEngine.from_args(retriever)

    summary_prompt = f"""
    Based on the following resume text, summarize the candidate's strongest technical skills, professional highlights,
    and key achievements. Write in 4-5 sentences, professional tone.
    Resume: {resume_text}
    """

    response = query_engine.query(summary_prompt)
    return response.response