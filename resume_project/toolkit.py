import subprocess
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.tools.wikipedia import WikipediaToolSpec
from dotenv import load_dotenv
import os
from additional_tool import InteractiveToolsManager

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Set up LlamaIndex with OpenAI
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=1024,
)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")  # Replace with get_openai_embedding_model() if available

wiki_tool_spec = WikipediaToolSpec()

try:
    tools_manager = InteractiveToolsManager()
except Exception as e:
    print(f"Error initializing InteractiveToolsManager: {str(e)}")
    tools_manager = None

def additional_general_knowledge_query(question: str) -> str:
    """Answer general knowledge questions using the LLM."""
    if tools_manager is None:
        return "InteractiveToolsManager not initialized."
    try:
        return tools_manager.general_knowledge_query(question)
    except Exception as e:
        return f"Error processing additional general knowledge query: {str(e)}"

additional_general_knowledge_tool = FunctionTool.from_defaults(
    fn=additional_general_knowledge_query,
    name="additional_general_knowledge_query",
    description="Answers general knowledge questions with comprehensive, accurate information."
)

def web_search_simulation(query: str) -> str:
    """Simulate a web search to provide current information on a topic."""
    if tools_manager is None:
        return "InteractiveToolsManager not initialized."
    try:
        return tools_manager.web_search_simulation(query)
    except Exception as e:
        return f"Error processing web search simulation: {str(e)}"

web_search_tool = FunctionTool.from_defaults(
    fn=web_search_simulation,
    name="web_search_simulation",
    description="Simulates a web search to provide current information, including multiple perspectives."
)

def get_candidate_statistics() -> str:
    """Retrieve statistical analysis of the candidate database."""
    if tools_manager is None:
        return "InteractiveToolsManager not initialized."
    try:
        return tools_manager.get_candidate_statistics()
    except Exception as e:
        return f"Error retrieving candidate statistics: {str(e)}"

candidate_stats_tool = FunctionTool.from_defaults(
    fn=get_candidate_statistics,
    name="get_candidate_statistics",
    description="Provides a comprehensive statistical analysis of the candidate database."
)

def interactive_conversation(user_message: str) -> str:
    """Handle conversational queries and follow-up questions."""
    if tools_manager is None:
        return "InteractiveToolsManager not initialized."
    try:
        return tools_manager.interactive_conversation(user_message)
    except Exception as e:
        return f"Error processing conversational query: {str(e)}"

conversation_tool = FunctionTool.from_defaults(
    fn=interactive_conversation,
    name="interactive_conversation",
    description="Handles complex conversational queries, follow-ups, or general discussions."
)

# Collect all tools
tools = [
    additional_general_knowledge_tool,
    web_search_tool,
    candidate_stats_tool,
    conversation_tool
]

# Build the ReAct agent
try:
    agent = ReActAgent(
        llm=Settings.llm,
        tools=tools,
        verbose=True,
        max_iterations=10
    )
except Exception as e:
    print(f"Error initializing ReActAgent: {str(e)}")
    agent = None

# User interface choice
def main():
    print("Welcome to the AI Assistant!")
    print("Choose an interface:")
    print("1. Interactive Terminal")
    print("2. Streamlit UI")
    
    choice = input("Enter number (1 or 2): ")
    
    if choice == "1":
        try:
            subprocess.run(["python", "resume_project/additional_tool.py"])
        except Exception as e:
            print(f"Error running interactive terminal: {str(e)}")
    elif choice == "2":
        try:
            subprocess.run(["streamlit", "run", "resume_project/streamlit_chatbot.py"])
        except Exception as e:
            print(f"Error launching Streamlit UI: {str(e)}")

if __name__ == "__main__":
    main()