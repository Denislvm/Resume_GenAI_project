import pandas as pd
from llama_index.core import Document

def load_data(file_path: str, limit: int = 30):
    df = pd.read_csv(file_path, encoding="utf-8")
    df_subset = df.head(limit)

    documents = []
    for _, row in df_subset.iterrows():
        resume_text = f"""
        Resume_str: {row.get('Resume_str', '')}
        Resume_html: {row.get('Resume_html', '')}
        Category: {row.get('Category', '')}
        """
        documents.append(Document(text=resume_text.strip()))

    # Filter by keywords
    keywords = ["HR", "Administrator", "Specialist", "Director"]
    filtered_docs = [
        doc for doc in documents
        if any(keyword.lower() in doc.text.lower() for keyword in keywords)
    ]
    return filtered_docs
