import os
import json
import pandas as pd
import re
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Set up LlamaIndex with OpenAI
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=256,  # Increased for better extraction
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Updated Pydantic model for extracted candidate info
class CandidateInfo(BaseModel):
    name: str
    profession: str
    years_experience: int

def smart_extract_name(text, candidate_id):
    """Improved name extraction focusing on first few lines only"""
    lines = text.split('\n')
    
    # Strategy 1: Look for name in first 5 lines only (header area)
    for i, line in enumerate(lines[:5]):
        line = line.strip()
        if not line or len(line) < 3 or len(line) > 40:  # Skip very long lines
            continue
        
        # Check if line looks like a standalone name
        if is_standalone_name(line):
            return line.title()
    
    # Strategy 2: Look for "Name:" patterns in first 10 lines
    for line in lines[:10]:
        name_match = re.search(r'(?:Name|Full\s+Name)\s*:\s*([A-Za-z\s]+)', line, re.I)
        if name_match:
            potential_name = name_match.group(1).strip().title()
            if is_valid_name_format(potential_name):
                return potential_name
    
    # Strategy 3: Extract from email (more conservative)
    email_match = re.search(r'\b([a-z]+)\.([a-z]+)@[a-z]+\.[a-z]+', text.lower())
    if email_match:
        first, last = email_match.groups()
        if len(first) > 1 and len(last) > 1:  # Avoid single letter names
            return f"{first.capitalize()} {last.capitalize()}"
    
    return "Unknown"

def is_standalone_name(line):
    """Check if a line is likely just a person's name"""
    line = line.strip()
    
    # Remove common prefixes
    line = re.sub(r'^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s*', '', line, flags=re.I)
    
    words = line.split()
    
    # Should be 2-4 words for a typical name
    if not (2 <= len(words) <= 4):
        return False
    
    # Each word should be properly capitalized and contain only letters
    for word in words:
        if not word.isalpha() or not (word.istitle() or word.isupper()):
            return False
        # Avoid very short or very long words
        if len(word) < 2 or len(word) > 15:
            return False
    
    # Check for job-related terms (stronger exclusions)
    job_terms = {
        'manager', 'engineer', 'developer', 'analyst', 'specialist', 'director',
        'coordinator', 'assistant', 'associate', 'senior', 'junior', 'lead',
        'principal', 'chief', 'operations', 'marketing', 'sales', 'human',
        'resources', 'financial', 'project', 'business', 'technical', 'software',
        'data', 'system', 'network', 'security', 'quality', 'customer', 'service',
        'administrative', 'executive', 'professional', 'consultant', 'officer',
        'supervisor', 'administrator', 'representative', 'technician'
    }
    
    line_lower = line.lower()
    if any(term in line_lower for term in job_terms):
        return False
    
    return True

def is_valid_name_format(name):
    """Simple validation for extracted names"""
    if not name or len(name) < 3:
        return False
    
    words = name.split()
    if len(words) < 2:
        return False
    
    # Should contain only letters and spaces
    if not all(c.isalpha() or c.isspace() for c in name):
        return False
    
    return True

def fallback_extract_profession(text):
    """Improved profession extraction"""
    lines = text.split('\n')[:15]  # Focus on header area
    
    # Common job title patterns
    title_patterns = [
        r'\b(?:Senior|Junior|Lead|Principal|Chief)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\b([A-Z][a-z]+\s+(?:Manager|Engineer|Developer|Analyst|Specialist|Director|Coordinator))\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:Manager|Engineer|Developer)\b',
    ]
    
    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
            
        for pattern in title_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if is_valid_profession(match):
                    return match
    
    # Fallback to simple capitalized phrases
    for line in lines:
        words = line.strip().split()
        if len(words) == 2 and all(w.istitle() for w in words):
            potential = ' '.join(words)
            if is_valid_profession(potential):
                return potential
    
    return "Professional"

def is_valid_profession(profession):
    """Check if text looks like a valid job title"""
    exclude_terms = {
        'summary', 'objective', 'skills', 'education', 'experience',
        'contact', 'phone', 'email', 'address', 'resume'
    }
    return not any(term in profession.lower() for term in exclude_terms)

def extract_years_experience(text):
    """More accurate years of experience extraction"""
    
    # Strategy 1: Look for explicit experience statements
    exp_patterns = [
        r'\b(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|work|professional)',
        r'(?:experience|work|professional).*?(\d+)\+?\s*years?',
        r'\b(\d+)\+?\s*years?\s+(?:in|of|working)',
        r'over\s+(\d+)\s+years?',
        r'more than\s+(\d+)\s+years?'
    ]
    
    for pattern in exp_patterns:
        matches = re.findall(pattern, text, re.I)
        if matches:
            years = [int(match) for match in matches if match.isdigit()]
            if years:
                return max(years)  # Take the highest mentioned years
    
    # Strategy 2: Calculate from employment dates
    current_year = datetime.now().year
    
    # Look for date ranges in format "YYYY - YYYY" or "YYYY - Present"
    date_ranges = re.findall(r'\b(\d{4})\s*[-–]\s*(?:(\d{4})|(?:present|current|now))\b', text, re.I)
    
    total_experience = 0
    for start_str, end_str in date_ranges:
        try:
            start_year = int(start_str)
            end_year = current_year if not end_str or end_str == '' else int(end_str)
            
            # Validate years are reasonable
            if 1990 <= start_year <= current_year and start_year <= end_year <= current_year:
                experience = end_year - start_year
                # Only count if it's a reasonable job duration (6 months to 40 years)
                if 0.5 <= experience <= 40:
                    total_experience += experience
        except ValueError:
            continue
    
    if total_experience > 0:
        return int(round(total_experience))
    
    # Strategy 3: Look for month-year ranges (more detailed dates)
    month_year_pattern = r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\s*[-–]\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})|(?:present|current))\b'
    month_ranges = re.findall(month_year_pattern, text, re.I)
    
    for start_str, end_str in month_ranges:
        try:
            start_year = int(start_str)
            end_year = current_year if not end_str or end_str == '' else int(end_str)
            
            if 1990 <= start_year <= current_year and start_year <= end_year <= current_year:
                experience = end_year - start_year
                if 0 <= experience <= 40:
                    total_experience += experience
        except ValueError:
            continue
    
    return int(round(total_experience)) if total_experience > 0 else 0

# Load Resume.csv
csv_path = "Resume.csv"
df = pd.read_csv(csv_path)
df = df.sample(n=5, random_state=42)  # 5 random resumes

# Convert CSV rows to LlamaIndex Documents
documents = []
for idx, row in df.iterrows():
    candidate_id = f"candidate_{idx}"
    documents.append(Document(
        text=row["Resume_str"],
        metadata={
            "candidate_id": candidate_id
        }
    ))

# Set up extraction program with improved prompt
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception)
)
def extract_with_retry(program, resume_text):
    return program(resume_text=resume_text)

extract_program = LLMTextCompletionProgram.from_defaults(
    output_cls=CandidateInfo,
    prompt_template_str=(
        "Carefully extract the following information from the resume:\n"
        "1. FULL NAME: Look ONLY at the first 1-3 lines of the resume. The name should be a standalone line with just the person's name (like 'John Smith' or 'JOHN SMITH'). Ignore any job titles or company names.\n"
        "2. PROFESSION: The main job title or role, usually found near the top or in a summary section.\n" 
        "3. YEARS OF EXPERIENCE: Look for phrases like 'X years of experience' OR calculate from employment date ranges (e.g., '2010-2015' = 5 years). Sum up all work experience periods.\n\n"
        "IMPORTANT: For names, be very strict - if you see job titles like 'Operations Manager' or company names, return 'Unknown' instead.\n\n"
        "If you cannot find clear information, use these defaults:\n"
        "- name: 'Unknown'\n"
        "- profession: 'Professional' \n"
        "- years_experience: 0\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        '{{"name": "value", "profession": "value", "years_experience": number}}\n\n'
        "Resume text:\n{resume_text}"
    ),
    llm=Settings.llm,
)

# Process each document
candidates = []
full_texts = {}

for doc in documents:
    print(f"\nProcessing {doc.metadata['candidate_id']}...")
    
    try:
        # Try LLM extraction first
        info = extract_with_retry(extract_program, doc.text[:3000])
        
        # Use LLM results or fallback to regex
        name = info.name if info.name != 'Unknown' else smart_extract_name(doc.text, doc.metadata['candidate_id'])
        profession = info.profession if info.profession != 'Professional' else fallback_extract_profession(doc.text)
        years_experience = info.years_experience if info.years_experience > 0 else extract_years_experience(doc.text)
        
    except Exception as e:
        print(f"LLM extraction failed for {doc.metadata['candidate_id']}: {e}")
        # Fall back to regex-only extraction
        name = smart_extract_name(doc.text, doc.metadata['candidate_id'])
        profession = fallback_extract_profession(doc.text)
        years_experience = extract_years_experience(doc.text)
    
    # Update document metadata
    doc.metadata.update({
        "name": name,
        "profession": profession,
        "years_experience": years_experience
    })
    
    candidate_dict = {
        "id": doc.metadata["candidate_id"],
        "name": name,
        "profession": profession,
        "years": years_experience,
    }
    
    candidates.append(candidate_dict)
    full_texts[candidate_dict["id"]] = doc.text
    
    print(f"Extracted - Name: {name}, Profession: {profession}, Years: {years_experience}")

# Save extracted data
with open("candidates.json", "w") as f:
    json.dump(candidates, f, indent=2)
with open("full_texts.json", "w") as f:
    json.dump(full_texts, f, indent=2)

print(f"\nSaved {len(candidates)} candidates to candidates.json")

# Split documents into chunks
node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

# Set up ChromaDB vector store
chroma_client = chromadb.PersistentClient(path="./vector_store")
chroma_collection = chroma_client.get_or_create_collection("resumes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build and persist the index
index = VectorStoreIndex(nodes, storage_context=storage_context)
print("Processing complete. Vector index built and stored in ChromaDB.")

# Print final results for verification
print("\n" + "="*50)
print("EXTRACTION RESULTS:")
print("="*50)
for candidate in candidates:
    print(f"ID: {candidate['id']}")
    print(f"Name: {candidate['name']}")
    print(f"Profession: {candidate['profession']}")
    print(f"Years Experience: {candidate['years']}")
    print("-" * 30)