import pandas as pd
from llama_index.core import Document

def load_data(file_path: str, limit: int = 30):
    """
    Load resumes from CSV and return them as LlamaIndex Document objects.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    if limit:
        df = df.sample(limit)

    documents = []
    for _, row in df.iterrows():
        resume_text = f"""
        Candidate ID: {row.get('ID', '')}
        Resume: {row.get('Resume_str', '')}
        Category: {row.get('Category', '')}
        """
        # Store category and ID as metadata for filtering and grouping
        documents.append(Document(
            text=resume_text.strip(),
            metadata={"category": row.get('Category', ''), "candidate_id": str(row.get('ID', ''))}
        ))

    return documents