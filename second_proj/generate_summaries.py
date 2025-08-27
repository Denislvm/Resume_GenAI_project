import json
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import chromadb
from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in .env file. Continuing without it.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Set up LlamaIndex
try:
    if openai_api_key:
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=512,
        )
    else:
        raise ValueError("No OpenAI API key provided.")
except Exception as e:
    print(f"Failed to initialize OpenAI LLM: {e}. Falling back to microsoft/phi-2.")
    Settings.llm = HuggingFaceLLM(
        model_name="microsoft/phi-2",
        tokenizer_name="microsoft/phi-2",
        model=AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ),
        tokenizer=AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True),
        context_window=2048,
        max_new_tokens=512,
    )

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector index
chroma_client = chromadb.PersistentClient(path="./vector_store")
chroma_collection = chroma_client.get_collection("resumes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

# Load pre-extracted data
candidates = json.load(open("candidates.json"))

# Generate summaries
summaries = {}
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception)
)
def query_with_retry(query_engine, query_str):
    return query_engine.query(query_str)

for candidate in candidates:
    candidate_id = candidate["id"]
    try:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="candidate_id", value=candidate_id)])
        query_engine = index.as_query_engine(filters=filters)
        response = query_with_retry(
            query_engine,
            "Based on the resume chunks, generate a concise summary of the candidate's strongest skills and professional highlights in 100-200 words."
        )
        summaries[candidate_id] = response.response if response.response.strip() else "No summary generated (LLM limitation)."
    except Exception as e:
        print(f"Error generating summary for {candidate_id}: {e}")
        summaries[candidate_id] = "Error generating summary. See full resume."

# Save summaries
with open("summaries.json", "w") as f:
    json.dump(summaries, f)

print("Summaries generated and saved to summaries.json.")