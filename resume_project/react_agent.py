import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import chromadb

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Set up LlamaIndex
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=1024,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

class AnonymousResumeRetrievalSystem:
    """Retrieval system that works with anonymous candidates using IDs and content."""
    
    def __init__(self):
        """Initialize the retrieval system."""
        print("Loading vector index...")
        # Load vector index
        chroma_client = chromadb.PersistentClient(path="./vector_store")
        chroma_collection = chroma_client.get_collection("resumes")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            storage_context=storage_context
        )
        
        print("Loading candidate data...")
        # Load pre-extracted data
        with open("candidates.json", "r") as f:
            self.candidates = json.load(f)
        with open("full_texts.json", "r") as f:
            self.full_texts = json.load(f)
        with open("summaries.json", "r") as f:
            self.summaries = json.load(f)
        
        # Create friendly identifiers for candidates
        self.candidate_labels = {}
        for i, candidate in enumerate(self.candidates):
            # Create a friendly label based on profession and experience
            profession = candidate.get('profession', 'Professional')
            years = candidate.get('years', 0)
            
            if profession != 'Professional':
                label = f"Candidate #{i+1}: {profession} ({years} years)"
            else:
                label = f"Candidate #{i+1}: {years} years experience"
            
            self.candidate_labels[candidate['id']] = {
                'label': label,
                'short_label': f"Candidate #{i+1}",
                'index': i+1
            }
        
        print(f"Loaded {len(self.candidates)} candidates successfully!")
        print("Candidate labels created for anonymous referencing.")

    def search_candidates_by_skills(self, query: str, top_k: int = 3) -> str:
        """Search for candidates based on skills and experience."""
        try:
            query_engine = self.index.as_query_engine(similarity_top_k=top_k * 3)
            response = query_engine.query(
                f"Find candidates with skills or experience related to: {query}"
            )
            
            # Extract candidate IDs from response
            candidate_ids = set()
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes[:top_k * 2]:  # Get more nodes
                    if 'candidate_id' in node.metadata:
                        candidate_ids.add(node.metadata['candidate_id'])
            
            # If no specific matches, search by keywords in profession/summary
            if not candidate_ids:
                query_lower = query.lower()
                for candidate in self.candidates:
                    profession_match = query_lower in candidate.get('profession', '').lower()
                    summary_match = query_lower in self.summaries.get(candidate['id'], '').lower()
                    if profession_match or summary_match:
                        candidate_ids.add(candidate['id'])
            
            # Format results
            results = []
            for candidate in self.candidates:
                if candidate['id'] in candidate_ids and len(results) < top_k:
                    summary = self.summaries.get(candidate['id'], 'No summary available')
                    label = self.candidate_labels[candidate['id']]['label']
                    
                    results.append(
                        f"✅ **{label}**\n"
                        f"   🎯 Summary: {summary[:200]}...\n"
                    )
            
            if not results:
                return f"❌ No candidates found matching '{query}'. Try broader terms like 'manager', 'technical', 'customer service', etc."
                
            return f"🔍 Found {len(results)} candidates matching '{query}':\n\n" + "\n" + "-"*50 + "\n".join(results)
            
        except Exception as e:
            return f"❌ Error searching candidates: {str(e)}"

    def get_candidate_by_reference(self, reference: str) -> str:
        """Get candidate details by various reference methods."""
        # Parse different reference formats
        candidate = None
        
        # Method 1: By number (e.g., "first", "second", "candidate 1", "#1")
        number_match = re.search(r'(?:candidate\s*#?|#)(\d+)', reference.lower())
        if number_match:
            try:
                idx = int(number_match.group(1)) - 1
                if 0 <= idx < len(self.candidates):
                    candidate = self.candidates[idx]
            except ValueError:
                pass
        
        # Method 2: By ordinal (first, second, third, etc.)
        ordinals = {
            'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4,
            '1st': 0, '2nd': 1, '3rd': 2, '4th': 3, '5th': 4
        }
        for word, idx in ordinals.items():
            if word in reference.lower() and idx < len(self.candidates):
                candidate = self.candidates[idx]
                break
        
        # Method 3: By profession/role
        if not candidate:
            for c in self.candidates:
                profession = c.get('profession', '').lower()
                if profession in reference.lower() and profession != 'professional':
                    candidate = c
                    break
        
        # Method 4: By experience level
        if not candidate:
            exp_match = re.search(r'(\d+)\s*years?', reference.lower())
            if exp_match:
                target_years = int(exp_match.group(1))
                for c in self.candidates:
                    if c.get('years', 0) == target_years:
                        candidate = c
                        break
        
        # Method 5: By superlatives (best, most experienced, etc.)
        if not candidate:
            if 'most experienced' in reference.lower() or 'highest experience' in reference.lower():
                candidate = max(self.candidates, key=lambda x: x.get('years', 0))
            elif 'least experienced' in reference.lower() or 'newest' in reference.lower():
                candidate = min(self.candidates, key=lambda x: x.get('years', 0))
            elif 'best' in reference.lower():
                # Return most experienced as "best"
                candidate = max(self.candidates, key=lambda x: x.get('years', 0))
        
        if not candidate:
            available_refs = [self.candidate_labels[c['id']]['label'] for c in self.candidates]
            return f"❌ Could not find candidate by '{reference}'. Try:\n" + "\n".join([f"• {ref}" for ref in available_refs])
        
        # Format detailed response
        label = self.candidate_labels[candidate['id']]['label']
        summary = self.summaries.get(candidate['id'], 'No summary available')
        full_text = self.full_texts.get(candidate['id'], 'Resume not available')
        
        result = f"""
👤 **{label}**

📋 **Profile Overview:**
• Profession: {candidate.get('profession', 'Not specified')}
• Experience: {candidate.get('years', 0)} years
• Candidate ID: {candidate['id']}

✨ **Professional Summary:**
{summary}

📄 **Resume Highlights:**
{self._extract_key_sections(full_text)}

💡 **Key Skills & Technologies:**
{self._extract_skills(full_text)}
        """
        
        return result.strip()
    
    def _extract_key_sections(self, text: str) -> str:
        """Extract key sections from resume text."""
        # Look for common resume sections
        sections = []
        lines = text.split('\n')
        
        current_section = ""
        for line in lines[:20]:  # First 20 lines usually contain key info
            line = line.strip()
            if len(line) > 10 and len(line) < 100:  # Reasonable length
                current_section += line + " "
        
        return current_section[:400] + "..." if current_section else "Resume content analysis not available."
    
    def _extract_skills(self, text: str) -> str:
        """Extract skills and technologies from resume."""
        skills = []
        common_skills = [
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node',
            'machine learning', 'data analysis', 'project management', 'leadership',
            'communication', 'microsoft office', 'excel', 'powerpoint', 'tableau',
            'customer service', 'sales', 'marketing', 'design', 'training'
        ]
        
        text_lower = text.lower()
        for skill in common_skills:
            if skill in text_lower:
                skills.append(skill.title())
        
        return ", ".join(skills[:8]) if skills else "Skills analysis not available from resume text."

    def rank_candidates_by_criteria(self, criteria: str) -> str:
        """Rank candidates by specific criteria."""
        if not self.candidates:
            return "❌ No candidates available."
        
        criteria_lower = criteria.lower()
        
        if 'experience' in criteria_lower:
            sorted_candidates = sorted(self.candidates, key=lambda x: x.get('years', 0), reverse=True)
            title = "📊 **Candidates Ranked by Experience Level:**"
        elif 'manager' in criteria_lower or 'management' in criteria_lower:
            # Filter and rank management roles
            mgmt_candidates = [c for c in self.candidates if 'manager' in c.get('profession', '').lower()]
            sorted_candidates = sorted(mgmt_candidates, key=lambda x: x.get('years', 0), reverse=True)
            if not mgmt_candidates:
                return "❌ No management candidates found in the pool."
            title = "📊 **Management Candidates Ranked by Experience:**"
        else:
            # Default ranking by experience
            sorted_candidates = sorted(self.candidates, key=lambda x: x.get('years', 0), reverse=True)
            title = f"📊 **Candidates Ranked by '{criteria}':**"
        
        results = title + "\n\n"
        
        for i, candidate in enumerate(sorted_candidates, 1):
            label = self.candidate_labels[candidate['id']]['label']
            summary = self.summaries.get(candidate['id'], 'No summary available')
            results += f"{i}. 🏆 **{label}**\n"
            results += f"   🎯 Key strengths: {summary[:150]}...\n\n"
        
        return results

    def find_best_candidate_for_role(self, role_description: str) -> str:
        """Find the best candidate for a specific role."""
        try:
            query_engine = self.index.as_query_engine(similarity_top_k=len(self.candidates))
            response = query_engine.query(
                f"Which candidate would be best suited for this role: {role_description}. "
                f"Consider their skills, experience, and background."
            )
            
            # Get similarity scores for all candidates
            candidate_scores = {}
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    if 'candidate_id' in node.metadata:
                        candidate_id = node.metadata['candidate_id']
                        score = getattr(node, 'score', 0)
                        if candidate_id not in candidate_scores or score > candidate_scores[candidate_id]:
                            candidate_scores[candidate_id] = score
            
            if not candidate_scores:
                return "❌ Could not analyze candidates for this role."
            
            # Sort by score
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            
            results = f"🎯 **Best Candidates for: '{role_description}'**\n\n"
            
            for i, (candidate_id, score) in enumerate(sorted_candidates[:3], 1):
                candidate = next(c for c in self.candidates if c['id'] == candidate_id)
                label = self.candidate_labels[candidate_id]['label']
                summary = self.summaries.get(candidate_id, 'No summary available')
                
                results += f"{i}. 🏆 **{label}** (Match: {score:.2f})\n"
                results += f"   💼 {candidate.get('profession', 'Professional')}\n"
                results += f"   📅 {candidate.get('years', 0)} years experience\n"
                results += f"   🎯 Why: {summary[:200]}...\n\n"
            
            return results
            
        except Exception as e:
            return f"❌ Error analyzing candidates for role: {str(e)}"

    def get_statistics(self) -> str:
        """Get comprehensive statistics."""
        if not self.candidates:
            return "❌ No candidates available."
        
        total_candidates = len(self.candidates)
        experiences = [c.get('years', 0) for c in self.candidates]
        avg_experience = sum(experiences) / total_candidates
        max_exp = max(experiences)
        min_exp = min(experiences)
        
        professions = {}
        for candidate in self.candidates:
            profession = candidate.get('profession', 'Unknown')
            professions[profession] = professions.get(profession, 0) + 1
        
        result = f"""
📊 **Anonymous Candidate Pool Analysis:**

👥 **Overview:**
• Total Candidates: {total_candidates}
• Average Experience: {avg_experience:.1f} years
• Experience Range: {min_exp} - {max_exp} years

📋 **Role Distribution:**
"""
        
        for profession, count in sorted(professions.items(), key=lambda x: x[1], reverse=True):
            result += f"• {profession}: {count} candidate(s)\n"
        
        result += f"\n🏷️ **Quick References:**\n"
        for candidate in self.candidates:
            label = self.candidate_labels[candidate['id']]['short_label']
            profession = candidate.get('profession', 'Professional')
            years = candidate.get('years', 0)
            result += f"• {label}: {profession} ({years} years)\n"
        
        return result.strip()

class AnonymousReActAgent:
    """ReAct-like agent that works with anonymous candidates."""
    
    def __init__(self):
        """Initialize the agent."""
        self.retrieval_system = AnonymousResumeRetrievalSystem()
        self.llm = Settings.llm
    
    def route_query(self, user_input: str) -> str:
        """Route user query to appropriate function with better intent recognition."""
        user_input_lower = user_input.lower()
        
        # Enhanced routing logic
        if any(phrase in user_input_lower for phrase in [
            'find candidates', 'search for', 'who has experience in', 'candidates with',
            'experience in', 'skills in', 'background in'
        ]):
            return self.retrieval_system.search_candidates_by_skills(user_input)
        
        elif any(phrase in user_input_lower for phrase in [
            'tell me about', 'details about', 'information about', 'show me',
            'candidate #', 'first candidate', 'second candidate', 'third candidate',
            'most experienced', 'best candidate', 'least experienced'
        ]):
            return self.retrieval_system.get_candidate_by_reference(user_input)
        
        elif any(phrase in user_input_lower for phrase in [
            'rank', 'compare', 'order by', 'sort by', 'who is better',
            'ranking', 'comparison'
        ]):
            criteria = user_input.replace('rank', '').replace('compare', '').strip()
            return self.retrieval_system.rank_candidates_by_criteria(criteria or 'experience')
        
        elif any(phrase in user_input_lower for phrase in [
            'best for', 'suitable for', 'match for', 'fit for', 'role of',
            'position of', 'job of'
        ]):
            return self.retrieval_system.find_best_candidate_for_role(user_input)
        
        elif any(phrase in user_input_lower for phrase in [
            'statistics', 'stats', 'overview', 'summary', 'how many',
            'total', 'pool', 'all candidates'
        ]):
            return self.retrieval_system.get_statistics()
        
        else:
            # Use LLM for complex queries
            try:
                prompt = f"""
Analyze this user query about anonymous candidates: "{user_input}"

Determine the best action:
1. SEARCH - finding candidates with specific skills/qualifications
2. DETAILS - getting details about a specific candidate (by number, role, or description)
3. RANK - comparing or ranking candidates by criteria
4. MATCH - finding best candidates for a specific role/position
5. STATS - general statistics or overview

Respond with just the action name.
"""
                response = self.llm.complete(prompt)
                action = response.text.strip().upper()
                
                if 'SEARCH' in action:
                    return self.retrieval_system.search_candidates_by_skills(user_input)
                elif 'DETAILS' in action:
                    return self.retrieval_system.get_candidate_by_reference(user_input)
                elif 'RANK' in action:
                    return self.retrieval_system.rank_candidates_by_criteria(user_input)
                elif 'MATCH' in action:
                    return self.retrieval_system.find_best_candidate_for_role(user_input)
                else:
                    return self.retrieval_system.get_statistics()
                    
            except Exception:
                return self.retrieval_system.get_statistics()
    
    def chat(self, user_input: str) -> str:
        """Main chat interface."""
        if not user_input.strip():
            return "❓ Please ask me something about the candidates!"
        
        try:
            return self.route_query(user_input)
        except Exception as e:
            return f"❌ Error processing your request: {str(e)}"

def main():
    """Main function to run the anonymous candidate agent."""
    print("🚀 Initializing Anonymous Resume Analysis Agent...")
    
    try:
        # Check if files exist
        required_files = ["candidates.json", "full_texts.json", "summaries.json"]
        for file in required_files:
            if not os.path.exists(file):
                print(f"❌ Missing required file: {file}")
                print("Please run the resume processing and summary generation scripts first.")
                return
        
        # Initialize agent
        agent = AnonymousReActAgent()
        
        print("\n" + "="*70)
        print("🎯 ANONYMOUS RESUME ANALYSIS AGENT - READY!")
        print("="*70)
        print("Ask me anything about candidates without knowing their names!")
        print("\n📝 Example queries:")
        print("• 'Find candidates with management experience'")
        print("• 'Tell me about the first candidate'") 
        print("• 'Show me the most experienced candidate'")
        print("• 'Who is best for a manager position?'")
        print("• 'Rank candidates by experience'")
        print("• 'Compare all candidates'")
        print("• 'Show me statistics'")
        print("\nType 'help' for more examples or 'quit' to exit.")
        print("="*70)
        
        while True:
            user_input = input("\n💬 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Thank you for using the Anonymous Resume Analysis Agent!")
                break
            
            if user_input.lower() == 'help':
                help_text = """
🆘 **Help - Query Examples:**

🔍 **Search Queries:**
• "Find candidates with Python experience"
• "Who has management background?"
• "Show me technical candidates"

👤 **Candidate Details:**
• "Tell me about candidate #1"
• "Show me the first candidate"
• "Details about the most experienced person"

📊 **Comparisons & Rankings:**
• "Rank all candidates by experience"
• "Compare candidates"
• "Who has the most experience?"

🎯 **Best Match Queries:**
• "Who is best for a project manager role?"
• "Find the best candidate for customer service"
• "Match candidates to technical position"

📈 **Statistics:**
• "Show me candidate statistics"
• "Give me an overview of all candidates"
• "How many candidates do we have?"
                """
                print(help_text)
                continue
            
            if not user_input:
                print("❓ Please enter a question.")
                continue
            
            print("\n🤔 Processing...")
            response = agent.chat(user_input)
            print(f"\n🤖 Agent Response:\n{response}")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("\n🔍 Troubleshooting checklist:")
        print("1. ✅ Run resume processing script first")
        print("2. ✅ Generate summaries with generate_summaries.py") 
        print("3. ✅ Set OpenAI API key in .env file")
        print("4. ✅ Install dependencies: pip install llama-index chromadb")

if __name__ == "__main__":
    main()