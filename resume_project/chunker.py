from llama_index.core.node_parser import SentenceSplitter

def create_chunks(documents, chunk_size=512, chunk_overlap=50):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.get_nodes_from_documents(documents)
    return chunks
