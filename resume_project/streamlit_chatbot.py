import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Page configuration
st.set_page_config(
    page_title="Custom Data Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Custom Data Chatbot")
st.caption("Chat with your documents using LlamaIndex and OpenAI")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in .env file. Please create a .env file with your OpenAI API key.")
    st.stop()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Set up LlamaIndex with OpenAI
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=1024,
)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# File uploader
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=['txt', 'pdf', 'docx'],
    accept_multiple_files=True,
    help="Upload documents to chat with"
)

# Function to initialize the chat engine
@st.cache_resource
def initialize_chat_engine(_uploaded_files):
    # Save uploaded files temporarily
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in _uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
    
    # Load documents and create index
    try:
        documents = SimpleDirectoryReader(temp_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        chat_engine = index.as_chat_engine(
            chat_mode="condense_question",
            verbose=True
        )
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir)
        
        return chat_engine
    except Exception as e:
        st.error(f"Error initializing chat engine: {str(e)}")
        return None

# Initialize chat engine if files are uploaded
if uploaded_files:
    if st.session_state.chat_engine is None:
        with st.spinner("Processing documents..."):
            st.session_state.chat_engine = initialize_chat_engine(uploaded_files)
        
        if st.session_state.chat_engine:
            st.success("Documents processed successfully! You can now start chatting.")
            # Add initial system message
            if not st.session_state.messages:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hello! I've processed your documents. Ask me anything about them!"
                })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    if not uploaded_files:
        st.error("Please upload documents to chat with.")
        st.stop()
    
    if st.session_state.chat_engine is None:
        st.error("Chat engine not initialized. Please upload documents first.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_engine.chat(prompt)
                st.markdown(response.response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.response
                })
                
                # Show sources if available
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    with st.expander("Sources"):
                        for i, node in enumerate(response.source_nodes):
                            st.write(f"**Source {i+1}:**")
                            st.write(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                            if hasattr(node, 'metadata'):
                                st.write(f"*Metadata:* {node.metadata}")
                            st.write("---")
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Clear chat button
if st.button("Clear Chat", type="secondary"):
    st.session_state.messages = []
    st.rerun()

# Instructions
with st.expander("How to use"):
    st.markdown("""
    1. **Create a .env file** in your project directory with your OpenAI API key:
       ```
       OPENAI_API_KEY=your_api_key_here
       ```
    2. **Upload your documents** (txt, pdf, docx files)
    3. **Wait for processing** - this may take a moment
    4. **Start chatting** with your documents!
    
    The chatbot uses GPT-3.5-turbo with temperature 0.3 and max 1024 tokens.
    It will answer questions based on the content of your uploaded documents.
    """)