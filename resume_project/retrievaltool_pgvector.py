import os
import psycopg2
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
os.environ["OPENAI_API_KEY"] = openai_api_key

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
        except psycopg2.Error as e:
            raise ValueError(f"Database connection error: {e}")

    def vector_search(self, query: str, limit: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Perform vector similarity search with relevance filtering."""
        try:
            query_embedding = self.embed_model.get_text_embedding(query)
            query_lower = query.lower()
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
            # Filter results for relevance
            filtered_results = []
            for row in results:
                profession = row[2] or ""
                content = row[4] or ""
                similarity_score = float(row[5])
                related_terms = [query_lower]
                if "designer" in query_lower:
                    related_terms.extend(["ui", "ux", "graphic", "interaction"])
                if any(term in profession.lower() or term in content.lower() for term in related_terms) and similarity_score < similarity_threshold:
                    filtered_results.append({
                        'candidate_id': row[0],
                        'name': row[1],
                        'profession': profession,
                        'years_experience': row[3],
                        'content': content or "No detailed resume content available.",
                        'similarity_score': similarity_score
                    })
            return filtered_results
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
                    'content': row[4] or "No detailed resume content available."
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
                            'content': result[4] or "No detailed resume content available."
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
                    'content': row[4] or "No detailed resume content available."
                }
                for row in results
            ]
        except Exception as e:
            print(f"Error getting most experienced: {e}")
            return []

retrieval_tool = ResumeRetrievalTool()

vector_search_tool = FunctionTool.from_defaults(
    fn=retrieval_tool.vector_search,
    name="vector_search",
    description="""
    Perform semantic similarity search on resumes using vector embeddings, filtering for relevance to the query term.
    Input: query (str) - the search query (e.g., 'designer'), limit (int, default=5), similarity_threshold (float, default=0.3).
    Output: List of dictionaries with candidate details, filtered to match the query term or synonyms (e.g., 'ui', 'ux' for 'designer') in profession or content.
    Use this for queries about specific skills or roles, e.g., 'designer', 'developer'.
    """
)

get_all_candidates_tool = FunctionTool.from_defaults(
    fn=retrieval_tool.get_all_candidates,
    name="get_all_candidates",
    description="""
    Retrieve a list of all candidates sorted by name.
    Input: limit (int, default=20) - maximum number of candidates to return.
    Output: List of dictionaries with candidate details.
    Use this only when the user explicitly asks to see all candidates, e.g., 'show all candidates'.
    """
)

get_candidate_by_position_tool = FunctionTool.from_defaults(
    fn=retrieval_tool.get_candidate_by_position,
    name="get_candidate_by_position",
    description="""
    Get a specific candidate by their position (1-indexed) when sorted alphabetically by name.
    Input: position (int) - the position number (e.g., 1 for first, 2 for second, etc.).
    Output: Dictionary with candidate details or None if not found.
    Use this for queries like 'show me the first candidate', 'fourth candidate', or 'give me detail info about the fourth candidate'.
    """
)

get_most_experienced_tool = FunctionTool.from_defaults(
    fn=retrieval_tool.get_most_experienced,
    name="get_most_experienced",
    description="""
    Get the top candidates with the most years of experience.
    Input: limit (int, default=1) - number of top candidates to return.
    Output: List of dictionaries with candidate details, including full resume content if available.
    Use this for queries about the most experienced candidates, e.g., 'most experienced candidate'.
    """
)

# Updated agent context to prioritize positional queries
agent_context = """
You are a helpful assistant for analyzing resumes stored in a PostgreSQL database.
Your goal is to answer user queries about candidates accurately by using the provided tools to retrieve data from the database.
Do not make up, assume, or hallucinate any candidate information—base your responses solely on the data retrieved from the tools.
Follow these guidelines:
1. For queries mentioning positional terms like 'first', 'second', 'third', 'fourth', etc. (e.g., 'fourth candidate', 'give me detail info about the fourth candidate'), use the `get_candidate_by_position` tool with the specified position number.
2. For queries about specific roles, skills, or professions (e.g., 'designer', 'developer'), use the `vector_search` tool with the query term. Filter results to ensure the query term or synonyms (e.g., 'ui', 'ux' for 'designer') appear in the profession or content.
3. If `vector_search` returns no results or no relevant matches (e.g., query term not in profession or content), respond with: 'No candidates found matching [query term] in the database.'
4. Use `get_all_candidates` only when the user explicitly requests all candidates, e.g., 'show all candidates', 'list all resumes'.
5. Use `get_most_experienced` only for queries explicitly about the most experienced candidates, e.g., 'most experienced candidate', 'top experienced resumes'.
6. If a tool call fails or no tool applies, respond: 'I cannot retrieve the requested information. Please rephrase your query.'
7. For queries containing 'detail info', include the full 'content' field from the database without truncation, regardless of the tool used.
8. For general conversational queries not requiring database access, respond naturally and concisely.
9. Format responses concisely:
   - For standard queries: Include name, profession, years of experience, and a brief summary of content (max 50 words).
   - For 'detail info' queries: Include name, profession, years of experience, and the full 'content' field without truncation.
   - For multiple candidates: Number each candidate and provide name, profession, years of experience, and a brief summary.
   - If no candidates are found, state clearly: 'No candidates found for [query term or position] in the database.'
10. Tool names are case-sensitive: `vector_search`, `get_all_candidates`, `get_candidate_by_position`, `get_most_experienced`.
11. Avoid excessive iterations by selecting the appropriate tool immediately based on the query.
12. For debugging, log retrieved data to ensure it matches expectations, but do not include logs in the final response unless requested.
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
    verbose=False,
    max_iterations=25,
    context=agent_context
)

def intelligent_candidate_query(query: str) -> str:
    """
    Main integration function that uses the ReAct agent to answer queries based on PostgreSQL data.
    Returns only the final answer, suppressing intermediate thought logs.
    """
    try:
        response = agent.chat(query)
        # For debugging: Log raw tool output if needed
        # print(f"Raw tool output for query '{query}': {response.source_nodes}")
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
    print("- 'Show me the fourth candidate'")
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