import os
import json
import psycopg2
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load env
load_dotenv()
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

PG_DATABASE = os.getenv("PG_DATABASE")
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")

# 2. Load resumes
with open("candidates.json") as f:
    candidates = json.load(f)
with open("full_texts.json") as f:
    full_texts = json.load(f)

# 3. Connect to Postgres
conn = psycopg2.connect(
    dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD,
    host=PG_HOST, port=PG_PORT
)
cur = conn.cursor()

# 4. Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    id SERIAL PRIMARY KEY,
    candidate_id TEXT UNIQUE,
    name TEXT,
    profession TEXT,
    years_experience INT,
    content TEXT,
    embedding VECTOR(1536)  -- pgvector extension must be enabled
);
""")

# 5. Insert + update with embeddings
for candidate in candidates:
    cid = candidate["id"]
    text = full_texts[cid]

    # генерим вектор
    vector = embed_model.get_text_embedding(text)

    cur.execute(
        """
        INSERT INTO resumes (candidate_id, name, profession, years_experience, content, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (candidate_id) DO UPDATE
        SET name = EXCLUDED.name,
            profession = EXCLUDED.profession,
            years_experience = EXCLUDED.years_experience,
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding;
        """,
        (
            cid,
            candidate["name"],
            candidate["profession"],
            candidate["years"],
            text,
            vector,
        )
    )

conn.commit()
cur.close()
conn.close()

print(f"✅ Inserted {len(candidates)} resumes with embeddings into Postgres")
