import os
import psycopg2
from typing import Dict, Any, List
import asyncio
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# Custom system prompt (unchanged)
CUSTOM_REACT_PROMPT = """\
You are a resume retrieval assistant designed to answer queries about candidates by retrieving relevant information from a database. Your goal is to provide concise, accurate responses (2-3 sentences, max 60 words per candidate) focusing on key details like name, profession, years of experience, and core strengths. Use tools to fetch candidates when needed.

## Tools
You have access to the following tools:
{tool_desc}

## Instructions
- For queries like "show me all candidates" or "list all candidates," use the `retrieve_all_candidates` tool.
- For specific queries (e.g., "find candidates with Python skills" or "who is the best software engineer"), use the `retrieve_candidates` tool with appropriate query and top_k parameters.
- For queries referring to the "first candidate," "second candidate," etc., use the most recently retrieved candidates if available; otherwise, fetch a candidate using `retrieve_candidates` with a generic query.
- If no candidates are found, respond with "No relevant candidates found."
- When listing multiple candidates, include name, profession, years of experience, and a brief summary for each.
- Always format tool inputs as valid JSON without comments or extra spaces.

## Output Format
To use a tool, provide:
Be helpful, accurate, and concise. Use tools appropriately to retrieve candidate information.
"""

class ResumeRetrievalTool:
    def __init__(self):
        """Initialize the resume retrieval tool with database connection and models."""
        try:
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            os.environ["OPENAI_API_KEY"] = openai_api_key

            Settings.llm = OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=512,
            )
            self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

            self.db_params = {
                'dbname': os.getenv("PG_DATABASE"),
                'user': os.getenv("PG_USER"),
                'password': os.getenv("PG_PASSWORD"),
                'host': os.getenv("PG_HOST", "127.0.0.1"),
                'port': int(os.getenv("PG_PORT", 5432))
            }

            self.last_candidates = []
            self.last_candidate = None

            self._test_db_connection()
            self.agent = self._initialize_react_agent()
            print("ResumeRetrievalTool initialized successfully.")
        except Exception as e:
            print(f"Error initializing ResumeRetrievalTool: {e}")
            raise

    def _test_db_connection(self):
        """Test the database connection."""
        try:
            conn = self._get_db_connection()
            conn.close()
            print("Database connection test successful.")
        except Exception as e:
            print(f"Database connection test failed: {e}")
            raise

    def _get_db_connection(self):
        """Establish database connection."""
        try:
            return psycopg2.connect(**self.db_params)
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def _initialize_react_agent(self):
        """Initialize the ReAct agent with retrieval tools."""
        try:
            retrieve_candidates_tool = FunctionTool.from_defaults(
                fn=self.retrieve_candidates,
                name="retrieve_candidates",
                description="Retrieve the most relevant candidates from the vector database based on a query using vector similarity. Args: query (str): the search query, top_k (int, optional): number of candidates to return, default 1."
            )
            retrieve_all_candidates_tool = FunctionTool.from_defaults(
                fn=self.retrieve_all_candidates,
                name="retrieve_all_candidates",
                description="Retrieve all candidates from the database without filtering. Use when the query asks for all candidates. No arguments."
            )
            tools = [retrieve_candidates_tool, retrieve_all_candidates_tool]
            tool_desc = "\n\n".join([f"{tool.metadata.name}: {tool.metadata.description}" for tool in tools])
            system_prompt = CUSTOM_REACT_PROMPT.format(tool_desc=tool_desc)
            return ReActAgent(
                llm=Settings.llm,
                tools=tools,
                verbose=True,
                system_prompt=system_prompt
            )
        except Exception as e:
            print(f"Error initializing ReAct agent: {e}")
            raise

    def retrieve_candidates(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """Retrieve the most relevant candidates based on query using vector similarity."""
        try:
            query_embedding = self.embed_model.get_text_embedding(query)
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    candidate_id,
                    name,
                    profession,
                    years_experience,
                    content,
                    (embedding <=> %s::vector) as similarity_score
                FROM resumes
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, query_embedding, top_k))
            results = cur.fetchall()
            candidates = []
            for row in results:
                candidate = {
                    'candidate_id': row[0],
                    'name': row[1],
                    'profession': row[2],
                    'years_experience': row[3],
                    'content': row[4],
                    'similarity_score': float(row[5])
                }
                candidates.append(candidate)
            self.last_candidates = candidates
            if candidates:
                self.last_candidate = candidates[0]
            cur.close()
            conn.close()
            print(f"Retrieved {len(candidates)} candidates for query: {query}")
            return candidates
        except Exception as e:
            print(f"Error retrieving candidates: {e}")
            return []

    def retrieve_all_candidates(self) -> List[Dict[str, Any]]:
        """Retrieve all candidates from the database."""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    candidate_id,
                    name,
                    profession,
                    years_experience,
                    content,
                    0 as similarity_score
                FROM resumes;
            """)
            results = cur.fetchall()
            candidates = []
            for row in results:
                candidate = {
                    'candidate_id': row[0],
                    'name': row[1],
                    'profession': row[2],
                    'years_experience': row[3],
                    'content': row[4],
                    'similarity_score': float(row[5])
                }
                candidates.append(candidate)
            self.last_candidates = candidates
            if candidates:
                self.last_candidate = candidates[0]
            cur.close()
            conn.close()
            print(f"Retrieved {len(candidates)} candidates (all candidates).")
            return candidates
        except Exception as e:
            print(f"Error retrieving all candidates: {e}")
            return []

    def generate_concise_response(self, candidate: Dict[str, Any], question: str) -> str:
        """Generate a concise response (2-3 sentences) for a candidate based on the query."""
        try:
            prompt = f"""
            Provide a concise response (2-3 sentences, max 60 words) answering the question about the candidate.
            Focus on key information relevant to the question, including their name, profession, experience, and core strengths.
            Question: {question}
            Candidate: {candidate['name']}
            Profession: {candidate['profession']}
            Years of Experience: {candidate['years_experience']}
            Resume Content: {candidate['content'][:1500]}
            Response:
            """
            response = Settings.llm.complete(prompt)
            answer = response.text.strip()
            sentences = [s.strip() for s in answer.split('.') if s.strip()]
            if len(sentences) > 3:
                answer = '. '.join(sentences[:3]).strip() + '.'
            elif len(sentences) < 2:
                answer += ' They excel in their field.'
            return answer if answer.endswith('.') else answer + '.'
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def process_query(self, query: str) -> str:
        """Process a general user query to find relevant candidates and generate a response."""
        try:
            # Handle "all candidates" queries
            if any(phrase in query.lower() for phrase in ["all candidates", "list all candidates"]):
                candidates = self.retrieve_all_candidates()
                if not candidates:
                    return f"""
💬 Your question: {query}
🤔 Processing...
🤖 Agent Response: No candidates found in the database.
"""
                responses = []
                for i, candidate in enumerate(candidates, 1):
                    response = self.generate_concise_response(candidate, query)
                    responses.append(f"{i}. {response}")
                return f"""
💬 Your question: {query}
🤔 Processing...
🤖 Agent Response:
{'\n'.join(responses)}
"""
            # Handle "first candidate" queries
            if "first candidate" in query.lower() and self.last_candidate:
                response = self.generate_concise_response(self.last_candidate, query)
                return f"""
💬 Your question: {query}
🤔 Processing...
🤖 Agent Response: {response}
"""
            # Handle "second candidate," "third candidate," etc.
            import re
            match = re.match(r".*(second|third|\d+(?:st|nd|rd|th))\s+candidate", query.lower())
            if match and self.last_candidates:
                ordinal = match.group(1)
                ordinal_map = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5}
                if ordinal in ordinal_map:
                    index = ordinal_map[ordinal] - 1
                else:
                    try:
                        index = int(re.search(r'\d+', ordinal).group()) - 1
                    except:
                        index = -1
                if 0 <= index < len(self.last_candidates):
                    response = self.generate_concise_response(self.last_candidates[index], query)
                    return f"""
💬 Your question: {query}
🤔 Processing...
🤖 Agent Response: {response}
"""
                else:
                    return f"""
💬 Your question: {query}
🤔 Processing...
🤖 Agent Response: No candidate found at position {ordinal}.
"""
            # Fallback to ReAct agent
            print(f"Processing query with ReAct agent: {query}")
            try:
                # Try synchronous run first
                agent_response = self.agent.run(query)
                return f"""
💬 Your question: {query}
🤔 Processing...
🤖 Agent Response: {agent_response.response}
"""
            except RuntimeError as e:
                if "no running event loop" in str(e).lower():
                    # Fallback to async run
                    try:
                        async def run_agent():
                            return await self.agent.run(query)
                        agent_response = asyncio.run(run_agent())
                        return f"""
💬 Your question: {query}
🤔 Processing...
🤖 Agent Response: {agent_response.response}
"""
                    except Exception as async_e:
                        raise Exception(f"Async run failed: {str(async_e)}")
                else:
                    raise e
            except AttributeError:
                raise AttributeError("ReActAgent does not support 'run'. Please check available methods: {}".format(dir(self.agent)))
        except Exception as e:
            print(f"Error processing query: {e}")
            return f"""
💬 Your question: {query}
🤔 Processing...
🤖 Agent Response: Error processing query: {str(e)}
"""

def main():
    """Interactive tool to query candidates with a general question."""
    try:
        print("Welcome to the Resume Retrieval Tool!")
        retrieval_tool = ResumeRetrievalTool()
        while True:
            print("\nWhat would you like to know about the candidate(s)?")
            user_question = input("Question: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("Exiting the tool. Goodbye!")
                break
            if not user_question:
                print("Please enter a valid question.")
                continue
            output = retrieval_tool.process_query(user_question)
            print(output)
    except Exception as e:
        print(f"Error running main: {e}")

if __name__ == "__main__":
    main()