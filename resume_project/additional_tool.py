"""
Interactive Terminal Interface for Additional Tools
User can ask any questions they want through terminal input
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

class InteractiveToolsManager:
    """Manager for interactive tools that respond to user terminal input."""
    
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
    
    def general_knowledge_query(self, question: str) -> str:
        """Answer any general knowledge question using OpenAI."""
        try:
            prompt = f"""
You are a knowledgeable assistant that can answer questions on any topic.

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
            return f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question."
    
    def web_search_simulation(self, query: str) -> str:
        """Simulate web search for current information using OpenAI."""
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
    
    def get_candidate_statistics(self) -> str:
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
    
    def interactive_conversation(self, user_message: str) -> str:
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
    
    def route_question(self, user_input: str) -> str:
        """Route user input to the most appropriate tool."""
        user_input_lower = user_input.lower()
        
        # Check for candidate statistics requests
        if any(phrase in user_input_lower for phrase in [
            'statistics', 'stats', 'candidate data', 'database info', 
            'how many candidates', 'candidate overview'
        ]):
            return self.get_candidate_statistics()
        
        # Check for web search simulation requests
        elif any(phrase in user_input_lower for phrase in [
            'search for', 'latest', 'recent', 'current', 'news about', 
            'what\'s happening with', 'find information about'
        ]):
            return self.web_search_simulation(user_input)
        
        # Check for general knowledge questions
        elif any(phrase in user_input_lower for phrase in [
            'what is', 'how does', 'tell me about', 'explain', 
            'who was', 'when did', 'where is', 'why does'
        ]):
            return self.general_knowledge_query(user_input)
        
        # Default to interactive conversation
        else:
            return self.interactive_conversation(user_input)

def main():
    """Main interactive terminal interface."""
    print("=" * 60)
    print("INTERACTIVE AI ASSISTANT - TERMINAL INTERFACE")
    print("=" * 60)
    print("Ask me anything! I can help with:")
    print("• General knowledge questions")
    print("• Current events and information searches")
    print("• Candidate database statistics (if available)")
    print("• General conversation and complex questions")
    print("\nType 'quit', 'exit', or 'q' to end the session.")
    print("Type 'help' for more information.")
    print("=" * 60)
    
    # Initialize tools manager
    try:
        tools_manager = InteractiveToolsManager()
        print("AI Assistant initialized successfully!")
    except Exception as e:
        print(f"Error initializing AI Assistant: {e}")
        return
    
    print("\nReady for your questions!")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYour question: ").strip()
            
            # Handle exit commands
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\nThank you for using the AI Assistant! Goodbye!")
                break
            
            # Handle help command
            if user_input.lower() == 'help':
                help_text = """
HELP - How to use this AI Assistant:

TYPES OF QUESTIONS YOU CAN ASK:

1. GENERAL KNOWLEDGE:
   • "What is quantum physics?"
   • "Tell me about the Roman Empire"
   • "How does photosynthesis work?"
   • "Who was Albert Einstein?"

2. CURRENT EVENTS/SEARCH:
   • "Search for latest space missions"
   • "What's happening with climate change?"
   • "Find information about AI developments"
   • "Recent news about technology"

3. CANDIDATE STATISTICS (if data available):
   • "Show me candidate statistics"
   • "How many candidates do we have?"
   • "Give me database overview"

4. CONVERSATIONAL:
   • Any complex questions or discussions
   • Follow-up questions
   • Personal opinions or advice

Simply type your question and press Enter!
"""
                print(help_text)
                continue
            
            # Handle empty input
            if not user_input:
                print("Please enter a question or type 'help' for assistance.")
                continue
            
            # Process the question
            print("\nProcessing your question...")
            response = tools_manager.route_question(user_input)
            
            # Display response
            print("\nResponse:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing your question: {e}")
            print("Please try again or rephrase your question.")

if __name__ == "__main__":
    main()