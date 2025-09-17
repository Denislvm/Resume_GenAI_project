import os
import json
import psycopg2
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Initialize embedding model
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Database configuration
PG_DATABASE = os.getenv("PG_DATABASE")
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")

# Load JSON data
try:
    with open("candidates.json") as f:
        candidates = json.load(f)
    with open("full_texts.json") as f:
        full_texts = json.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not find JSON file: {e}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON format: {e}")
    exit(1)

# Connect to Postgres
try:
    conn = psycopg2.connect(
        dbname=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT
    )
    cur = conn.cursor()
except psycopg2.Error as e:
    print(f"Error: Could not connect to PostgreSQL: {e}")
    exit(1)

# Create table if not exists
try:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id SERIAL PRIMARY KEY,
            candidate_id TEXT UNIQUE,
            name TEXT,
            profession TEXT,
            years_experience INT,
            content TEXT,
            embedding VECTOR(1536) -- pgvector extension must be enabled
        );
    """)
except psycopg2.Error as e:
    print(f"Error: Could not create table: {e}")
    conn.rollback()
    cur.close()
    conn.close()
    exit(1)

# Clear existing data in the resumes table
try:
    cur.execute("TRUNCATE TABLE resumes RESTART IDENTITY;")
    print("✅ Cleared existing data in the resumes table")
except psycopg2.Error as e:
    print(f"Error: Could not truncate table: {e}")
    conn.rollback()
    cur.close()
    conn.close()
    exit(1)

# Insert new data
try:
    for candidate in candidates:
        cid = candidate["id"]
        text = full_texts.get(cid)
        if not text:
            print(f"Warning: No full text found for candidate ID {cid}, skipping")
            continue
        vector = embed_model.get_text_embedding(text)
        cur.execute(
            """
            INSERT INTO resumes (candidate_id, name, profession, years_experience, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (
                cid,
                candidate.get("name"),
                candidate.get("profession"),
                candidate.get("years"),
                text,
                vector,
            )
        )
    conn.commit()
    print(f"✅ Inserted {len(candidates)} resumes with embeddings into Postgres")
except psycopg2.Error as e:
    print(f"Error: Could not insert data: {e}")
    conn.rollback()
except Exception as e:
    print(f"Error: Unexpected error during data insertion: {e}")
    conn.rollback()
finally:
    cur.close()
    conn.close()
