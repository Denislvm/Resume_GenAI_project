import os
import psycopg2
from typing import Dict, Any, List
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

class ResumeRetrievalTool:
    def __init__(self):
        """Initialize the resume retrieval tool with database connection and models."""
        load_dotenv()
        
        # Initialize OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Set up LlamaIndex with OpenAI
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=512,
        )
        
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        # Database connection parameters
        self.db_params = {
            'dbname': os.getenv("PG_DATABASE"),
            'user': os.getenv("PG_USER"),
            'password': os.getenv("PG_PASSWORD"),
            'host': os.getenv("PG_HOST", "127.0.0.1"),
            'port': int(os.getenv("PG_PORT", 5432))
        }
        
        # Initialize ReAct agent with tools
        self.agent = self._initialize_react_agent()
    
    def _get_db_connection(self):
        """Establish database connection."""
        return psycopg2.connect(**self.db_params)
    
    def _initialize_react_agent(self):
        """Initialize the ReAct agent with retrieval tools."""
        # Define tool for retrieving candidates by query
        retrieve_candidates_tool = FunctionTool.from_defaults(
            fn=self.retrieve_candidates,
            name="retrieve_candidates",
            description="Retrieve the most relevant candidate from the vector database based on a query."
        )
        
        # Initialize ReAct agent
        return ReActAgent(
            llm=Settings.llm,
            tools=[retrieve_candidates_tool],
            verbose=True
        )
    
    def retrieve_candidates(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant candidate based on query using vector similarity.
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embed_model.get_text_embedding(query)
            
            # Connect to database
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            # Perform similarity search using cosine similarity
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
            
            # Format results
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
            
            cur.close()
            conn.close()
            
            return candidates
            
        except Exception as e:
            print(f"Error retrieving candidates: {e}")
            return []
    
    def generate_concise_response(self, candidate: Dict[str, Any], question: str) -> str:
        """
        Generate a concise response (2-3 sentences) for the candidate based on the query.
        """
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
            
            # Ensure 2-3 sentences
            sentences = answer.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) > 3:
                answer = '. '.join(sentences[:3]).strip() + '.'
            elif len(sentences) < 2:
                answer += ' They excel in their field.'
            
            return answer if answer.endswith('.') else answer + '.'
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """
        Process a general user query to find the most relevant candidate and return a concise response.
        """
        try:
            # Use ReAct agent to process general query
            candidates = self.retrieve_candidates(query, top_k=1)
            
            if not candidates:
                return "🤖 Agent Response: No relevant candidates found."
            
            # Process the top candidate
            candidate = candidates[0]
            answer = self.generate_concise_response(candidate, query)
            return f"""
💬 Your question: {query}
🤔 Processing...
🤖 Agent Response: {answer}
"""
                
        except Exception as e:
            return f"🤖 Agent Response: Error processing query: {str(e)}"

def main():
    """Interactive tool to query candidates with a general question."""
    retrieval_tool = ResumeRetrievalTool()
    
    while True:
        print("What would you like to know about the candidate(s)?")
        user_question = input("Question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_question:
            print("Please enter a valid question.")
            continue
            
        output = retrieval_tool.process_query(user_question)
        print(output)

if __name__ == "__main__":
    main()