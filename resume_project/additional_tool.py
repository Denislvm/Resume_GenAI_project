"""
Additional Tools for ReAct Agent Enhancement
Contains: General Knowledge Tool (using OpenAI) and Toolbox Integration
"""

import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

# Load environment and configure OpenAI
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

class AdditionalTools:
    """Container for additional tools to enhance the ReAct agent capabilities."""
    
    def __init__(self, candidates_file="candidates.json"):
        """Initialize with candidate data and OpenAI LLM."""
        self.candidates = self._load_candidates(candidates_file)
        self.llm = Settings.llm
    
    def _load_candidates(self, candidates_file: str) -> List[Dict]:
        """Load candidate data from JSON file."""
        try:
            with open(candidates_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {candidates_file} not found. Statistics tool will have limited functionality.")
            return []
        except Exception as e:
            print(f"Error loading candidates: {e}")
            return []
    
    def create_general_knowledge_tool(self) -> FunctionTool:
        """Create a true general knowledge tool using OpenAI for any topic unrelated to resumes."""
        
        def general_knowledge_query(question: str) -> str:
            """Answer general knowledge questions on any topic unrelated to resumes using OpenAI."""
            
            try:
                # Create a comprehensive prompt for general knowledge
                prompt = f"""
You are a knowledgeable assistant that can answer questions on any general topic. 
The user is asking about something unrelated to resumes or job applications.

Question: {question}

Please provide a comprehensive, accurate, and informative answer. Include:
- Key facts and explanations
- Historical context if relevant
- Current information if applicable
- Multiple perspectives if appropriate

Keep the response informative but concise (2-4 paragraphs maximum).
If you're not certain about specific details, acknowledge uncertainty.
"""
                
                response = self.llm.complete(prompt)
                return response.text.strip()
                
            except Exception as e:
                return f"I encountered an error while processing your question about '{question}': {str(e)}. Please try rephrasing your question."
        
        return FunctionTool.from_defaults(
            fn=general_knowledge_query,
            name="general_knowledge",
            description="Answer any general knowledge questions unrelated to resumes. This includes science, history, current events, entertainment, sports, technology, culture, arts, geography, politics, or any other general topic the user wants to know about."
        )
    
    def create_toolbox_integration_tool(self) -> List[FunctionTool]:
        """Create tools using Toolbox integration (placeholder for future implementation)."""
        tools = []
        
        try:
            # This is a placeholder for Toolbox integration
            # You would need to set up a Toolbox server first
            
            def toolbox_search(query: str) -> str:
                """Search using Toolbox integration (placeholder implementation)."""
                return f"Toolbox integration for '{query}' is not yet configured. Please set up a Toolbox server and update this implementation."
            
            toolbox_tool = FunctionTool.from_defaults(
                fn=toolbox_search,
                name="toolbox_search",
                description="Advanced search and data processing using Toolbox integration. Currently not configured."
            )
            
            # Uncomment when Toolbox server is set up:
            # from toolbox_llamaindex import ToolboxClient
            # 
            # async def load_toolbox_tools():
            #     async with ToolboxClient("http://127.0.0.1:5000") as client:
            #         return await client.aload_toolset()
            #
            # tools = asyncio.run(load_toolbox_tools())
            
            tools.append(toolbox_tool)
            
        except Exception as e:
            print(f"Toolbox integration not available: {e}")
            
        return tools
    
    def create_web_search_tool(self) -> FunctionTool:
        """Create a web search simulation tool using OpenAI for current information."""
        
        def web_search_simulation(query: str) -> str:
            """Simulate web search for current information using OpenAI's knowledge."""
            
            try:
                prompt = f"""
You are simulating a web search to answer current questions. The user is searching for: {query}

Based on your training data, provide information as if you found it from recent web sources. Include:
- Multiple relevant points about the topic
- Different perspectives or sources when applicable
- Acknowledge if information might be outdated due to your knowledge cutoff

Format your response as if presenting search results:
"Here's what I found about '{query}':"

Then provide 2-3 main points with explanations.

If the query is about very recent events (after your knowledge cutoff), acknowledge this limitation.
"""
                
                response = self.llm.complete(prompt)
                return response.text.strip()
                
            except Exception as e:
                return f"Search simulation failed for '{query}': {str(e)}"
        
        return FunctionTool.from_defaults(
            fn=web_search_simulation,
            name="web_search_simulation",
            description="Simulate web search for current information and recent topics using AI knowledge. Use when users ask for recent news, current events, or trending topics."
        )
    
    def create_candidate_statistics_tool(self) -> FunctionTool:
        """Create a tool for candidate database statistics and analytics."""
        
        def get_candidate_statistics() -> str:
            """Get comprehensive statistical analysis of the candidate database."""
            
            if not self.candidates:
                return "No candidate data available. Please ensure candidates.json exists and contains data."
            
            total_candidates = len(self.candidates)
            
            # Experience analysis
            experiences = [candidate.get('years', 0) for candidate in self.candidates]
            avg_experience = sum(experiences) / total_candidates if experiences else 0
            max_experience = max(experiences) if experiences else 0
            min_experience = min(experiences) if experiences else 0
            
            # Experience distribution
            experience_ranges = {
                "Entry Level (0-2 years)": 0,
                "Mid Level (3-5 years)": 0, 
                "Senior Level (6-10 years)": 0,
                "Expert Level (10+ years)": 0
            }
            
            for years in experiences:
                if years <= 2:
                    experience_ranges["Entry Level (0-2 years)"] += 1
                elif years <= 5:
                    experience_ranges["Mid Level (3-5 years)"] += 1
                elif years <= 10:
                    experience_ranges["Senior Level (6-10 years)"] += 1
                else:
                    experience_ranges["Expert Level (10+ years)"] += 1
            
            # Profession analysis
            professions = {}
            for candidate in self.candidates:
                profession = candidate.get('profession', 'Unknown')
                professions[profession] = professions.get(profession, 0) + 1
            
            # Most/least experienced candidates
            most_exp_candidate = max(self.candidates, key=lambda x: x.get('years', 0)) if self.candidates else None
            least_exp_candidate = min(self.candidates, key=lambda x: x.get('years', 0)) if self.candidates else None
            
            # Build comprehensive statistics report
            stats_report = f"""
CANDIDATE DATABASE STATISTICS
============================

OVERVIEW:
• Total Candidates: {total_candidates}
• Average Experience: {avg_experience:.1f} years
• Experience Range: {min_experience} - {max_experience} years

EXPERIENCE DISTRIBUTION:
"""
            for range_name, count in experience_ranges.items():
                percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
                stats_report += f"• {range_name}: {count} candidates ({percentage:.1f}%)\n"
            
            stats_report += f"\nPROFESSION BREAKDOWN:\n"
            sorted_professions = sorted(professions.items(), key=lambda x: x[1], reverse=True)
            for profession, count in sorted_professions:
                percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
                stats_report += f"• {profession}: {count} candidates ({percentage:.1f}%)\n"
            
            if most_exp_candidate and least_exp_candidate:
                stats_report += f"\nEXPERIENCE HIGHLIGHTS:\n"
                stats_report += f"• Most Experienced: {most_exp_candidate.get('profession', 'Unknown')} with {most_exp_candidate.get('years', 0)} years\n"
                stats_report += f"• Least Experienced: {least_exp_candidate.get('profession', 'Unknown')} with {least_exp_candidate.get('years', 0)} years\n"
            
            # Additional insights
            stats_report += f"\nINSIGHTS:\n"
            
            if avg_experience > 7:
                stats_report += "• High-experience candidate pool - good for senior positions\n"
            elif avg_experience < 3:
                stats_report += "• Entry-to-mid level candidate pool - good for growing teams\n"
            else:
                stats_report += "• Balanced experience levels across the candidate pool\n"
            
            if len(professions) > total_candidates * 0.7:
                stats_report += "• Diverse professional backgrounds represented\n"
            else:
                stats_report += "• Concentrated expertise in specific professional areas\n"
            
            return stats_report.strip()
        
        return FunctionTool.from_defaults(
            fn=get_candidate_statistics,
            name="candidate_statistics",
            description="Provide comprehensive statistical analysis of the candidate database including experience distribution, profession breakdown, and insights about the candidate pool."
        )
    
    def create_interactive_qa_tool(self) -> FunctionTool:
        """Create an interactive Q&A tool for complex conversations."""
        
        def interactive_qa(user_message: str) -> str:
            """Handle complex conversational queries and follow-up questions."""
            
            try:
                prompt = f"""
You are having a conversation with a user. They are asking: "{user_message}"

This appears to be a conversational question that might be:
- A follow-up to a previous topic
- A complex question requiring detailed explanation
- A request for clarification or elaboration
- A general discussion topic

Provide a helpful, engaging response that:
- Addresses their specific question or concern
- Provides relevant information and context
- Encourages further conversation if appropriate
- Is conversational and natural in tone

User message: {user_message}
"""
                
                response = self.llm.complete(prompt)
                return response.text.strip()
                
            except Exception as e:
                return f"I had trouble processing your message: {str(e)}. Could you please rephrase or ask in a different way?"
        
        return FunctionTool.from_defaults(
            fn=interactive_qa,
            name="interactive_conversation",
            description="Handle complex conversational queries, follow-up questions, and general discussion topics. Use when the user's message doesn't fit other specific tools but requires a thoughtful conversational response."
        )
    
    def get_all_tools(self) -> List[FunctionTool]:
        """Get all additional tools as a single list."""
        tools = []
        
        # Add general knowledge tool (primary)
        tools.append(self.create_general_knowledge_tool())
        
        # Add web search simulation tool
        tools.append(self.create_web_search_tool())
        
        # Add interactive Q&A tool
        tools.append(self.create_interactive_qa_tool())
        
        # Add candidate statistics tool
        tools.append(self.create_candidate_statistics_tool())
        
        # Add toolbox integration (placeholder)
        toolbox_tools = self.create_toolbox_integration_tool()
        tools.extend(toolbox_tools)
        
        return tools

def create_additional_tools(candidates_file="candidates.json") -> List[FunctionTool]:
    """
    Convenience function to create all additional tools.
    
    Args:
        candidates_file: Path to candidates JSON file for statistics tool
    
    Returns:
        List of additional tools for the ReAct agent
    """
    tool_creator = AdditionalTools(candidates_file)
    return tool_creator.get_all_tools()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Additional Tools with OpenAI...")
    
    # Create tools
    tool_creator = AdditionalTools()
    tools = tool_creator.get_all_tools()
    
    print(f"Created {len(tools)} additional tools:")
    for tool in tools:
        print(f"- {tool.metadata.name}: {tool.metadata.description}")
    
    # Test general knowledge tool
    print("\n" + "="*50)
    print("Testing General Knowledge Tool:")
    general_tool = tool_creator.create_general_knowledge_tool()
    
    test_questions = [
        "What is quantum physics?",
        "Tell me about the Roman Empire",
        "What are the latest developments in artificial intelligence?",
        "How does photosynthesis work?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        response = general_tool.fn(question)
        print(f"A: {response}")
        print("-" * 30)
    
    # Test web search simulation
    print("\n" + "="*50)
    print("Testing Web Search Simulation:")
    search_tool = tool_creator.create_web_search_tool()
    
    search_query = "latest space exploration missions"
    print(f"\nSearching: {search_query}")
    search_response = search_tool.fn(search_query)
    print(f"Results: {search_response}")
    
    # Test statistics tool  
    print("\n" + "="*50)
    print("Testing Statistics Tool:")
    stats_tool = tool_creator.create_candidate_statistics_tool()
    stats_response = stats_tool.fn()
    print(stats_response)