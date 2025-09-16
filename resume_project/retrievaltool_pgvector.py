import os
import json
import psycopg2
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
import re

# Load environment variables
load_dotenv()

# Setup OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Configure LlamaIndex settings
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1024,
)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Database configuration
PG_DATABASE = os.getenv("PG_DATABASE")
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")

class ResumeRetrievalTool:
    """Enhanced retrieval tool for resume database queries."""
    def __init__(self):
        self.embed_model = Settings.embed_model
        self.connection_params = {
            'dbname': PG_DATABASE,
            'user': PG_USER,
            'password': PG_PASSWORD,
            'host': PG_HOST,
            'port': PG_PORT
        }

    def _get_connection(self):
        """Get database connection with error handling."""
        try:
            return psycopg2.connect(**self.connection_params)
        except Exception as e:
            print(f"Database connection error: {e}")
            raise

    def vector_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        try:
            query_embedding = self.embed_model.get_text_embedding(query)
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            candidate_id,
                            name,
                            profession,
                            years_experience,
                            content,
                            (embedding <=> %s::vector) as similarity_score
                        FROM resumes
                        ORDER BY similarity_score ASC
                        LIMIT %s;
                    """, (query_embedding, limit))
                    results = cur.fetchall()
            return [
                {
                    'candidate_id': row[0],
                    'name': row[1],
                    'profession': row[2],
                    'years_experience': row[3],
                    'content': row[4],
                    'similarity_score': float(row[5])
                }
                for row in results
            ]
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def get_all_candidates(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all candidates with basic info."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT candidate_id, name, profession, years_experience, content
                        FROM resumes
                        ORDER BY name
                        LIMIT %s
                    """, (limit,))
                    results = cur.fetchall()
            return [
                {
                    'candidate_id': row[0],
                    'name': row[1],
                    'profession': row[2],
                    'years_experience': row[3],
                    'content': row[4]
                }
                for row in results
            ]
        except Exception as e:
            print(f"Error getting all candidates: {e}")
            return []

    def get_candidate_by_position(self, position: int) -> Optional[Dict[str, Any]]:
        """Get candidate by position (1-indexed)."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT candidate_id, name, profession, years_experience, content
                        FROM resumes
                        ORDER BY name
                        LIMIT 1 OFFSET %s
                    """, (position - 1,))
                    result = cur.fetchone()
            if result:
                return {
                    'candidate_id': result[0],
                    'name': result[1],
                    'profession': result[2],
                    'years_experience': result[3],
                    'content': result[4]
                }
            return None
        except Exception as e:
            print(f"Error getting candidate by position: {e}")
            return None

    def get_most_experienced(self, limit: int = 1) -> List[Dict[str, Any]]:
        """Get candidates with most experience."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT candidate_id, name, profession, years_experience, content
                        FROM resumes
                        ORDER BY years_experience DESC
                        LIMIT %s
                    """, (limit,))
                    results = cur.fetchall()
            return [
                {
                    'candidate_id': row[0],
                    'name': row[1],
                    'profession': row[2],
                    'years_experience': row[3],
                    'content': row[4]
                }
                for row in results
            ]
        except Exception as e:
            print(f"Error getting most experienced: {e}")
            return []

# Initialize components
retrieval_tool = ResumeRetrievalTool()

# Create individual tools for ReAct agent
vector_search_tool = FunctionTool.from_defaults(
    fn=retrieval_tool.vector_search,
    name="vector_search",
    description="""
    Perform semantic similarity search on resumes using vector embeddings.
    Input: query (str) - the search query, limit (int, default=5) - number of results.
    Output: List of dictionaries with candidate details including name, profession, years_experience, content, and similarity_score.
    Use this for queries about specific skills, experiences, or professions.
    """
)

get_all_candidates_tool = FunctionTool.from_defaults(
    fn=retrieval_tool.get_all_candidates,
    name="get_all_candidates",
    description="""
    Retrieve a list of all candidates sorted by name.
    Input: limit (int, default=20) - maximum number of candidates to return.
    Output: List of dictionaries with candidate details including name, profession, years_experience, content.
    Use this when the user asks to see all candidates.
    """
)

get_candidate_by_position_tool = FunctionTool.from_defaults(
    fn=retrieval_tool.get_candidate_by_position,
    name="get_candidate_by_position",
    description="""
    Get a specific candidate by their position (1-indexed) when sorted alphabetically by name.
    Input: position (int) - the position number.
    Output: Dictionary with candidate details or None if not found.
    Use this for queries like 'show me the first candidate' or 'fifth candidate'.
    """
)

get_most_experienced_tool = FunctionTool.from_defaults(
    fn=retrieval_tool.get_most_experienced,
    name="get_most_experienced",
    description="""
    Get the top candidates with the most years of experience.
    Input: limit (int, default=1) - number of top candidates to return.
    Output: List of dictionaries with candidate details.
    Use this for queries about the most experienced candidates.
    """
)

# Custom context for the ReAct agent to ensure accurate data retrieval
agent_context = """
You are a helpful assistant for analyzing resumes stored in a PostgreSQL database.
Your goal is to answer user queries about candidates accurately by always using the provided tools to retrieve data from the database.
Do not make up, assume, or hallucinate any candidate information—base your responses solely on the data retrieved from the tools.
If a query requires data retrieval, select and use the most appropriate tool (vector_search, get_all_candidates, get_candidate_by_position, get_most_experienced).
If no tool applies or if a tool call fails, inform the user that you cannot retrieve the information and suggest rephrasing the query.
For general conversational queries not requiring database access, respond naturally and concisely.
Summarize and present the retrieved data in a clear, concise, and natural way.
Tool names are case-sensitive and must be used exactly as defined.
Ensure the tool returns concise summaries or key information from the retrieved data. 
When presenting candidate data, format it clearly concise with name, profession, years of experience, and a brief summary of their content.
For multiple candidates, number each candidate and provide a brief summary for each.
"""

# Create ReAct agent with the tools and custom context
agent = ReActAgent.from_tools(
    tools=[
        vector_search_tool,
        get_all_candidates_tool,
        get_candidate_by_position_tool,
        get_most_experienced_tool
    ],
    llm=Settings.llm,
    verbose=False,  # Disable verbose output to hide thought process
    max_iterations=15,
    context=agent_context
)

def intelligent_candidate_query(query: str) -> str:
    """
    Main integration function that uses the ReAct agent to answer queries based on PostgreSQL data.
    Returns only the final answer, suppressing intermediate thought logs.
    """
    try:
        response = agent.chat(query)
        return str(response.response)
    except Exception as e:
        return f"I encountered an error while processing your query: {str(e)}. Please try again or rephrase your question."

def main():
    """Main interactive loop with intelligent answer generation."""
    print("🧠 Intelligent Resume Analysis System")
    print("=" * 45)
    print("Ask questions and get smart answers based on your PostgreSQL resume database!")
    print("\nExamples:")
    print("- 'Who is the most experienced candidate?'")
    print("- 'Show me the third candidate'")
    print("- 'Find me a candidate with design experience'")
    print("- 'Do you have any senior developers?'")
    print("- 'Tell me about candidates with administration experience'")
    print("\nType 'quit' to exit.\n")
    while True:
        try:
            user_query = input("🤔 Your question: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not user_query:
                print("Please ask a question about the candidates.")
                continue
            print("\n🤖 Analyzing candidates...")
            response = intelligent_candidate_query(user_query)
            print(f"\n💡 Answer:\n{response}\n")
            print("-" * 50)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ System Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()