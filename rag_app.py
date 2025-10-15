# Goal: Create a high-performance, dual-mode (RAG & Chatbot) AI application optimized for M1/M2/M3 Macs.
# Key Features: Metal GPU acceleration (MPS), Caching, Dark Mode UI, Multilingual support.

import streamlit as st
import tempfile
import os
import time
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# --- M1 Optimization Settings ---
# Note: Ensure ollama pull mistral is run and the model is built with the Modelfile: ollama create mistral-metal-accel -f Mistral-Metal.Modelfile
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral-metal-accel"


# --- Core Functions ---

@st.cache_resource
def get_embedding_function():
    """Loads and caches the embedding model, explicitly targeting the Metal GPU (MPS)."""
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        st.sidebar.success("✅ **Metal GPU (MPS) Detected:** Using GPU for Vectorization.")
    else:
        DEVICE = "cpu"
        st.sidebar.warning("⚠️ **Metal GPU NOT Detected:** Falling back to CPU for Vectorization.")

    # Use HuggingFaceEmbeddings as it allows explicit device assignment via PyTorch
    model_kwargs = {'device': DEVICE}
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)


@st.cache_resource(hash_funcs={UploadedFile: lambda x: x.file_id})
def load_and_process_docs(uploaded_file):
    """Caches the heavy document processing step based on the unique file hash."""
    st.sidebar.info("Processing documents (Slowest part on first run)...")
    start_time = time.time()

    # Safely write uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 1. Load, Split, and Embed
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        embeddings = get_embedding_function()
        vectorstore = FAISS.from_documents(texts, embeddings)

        # 2. Record Metrics
        end_time = time.time()
        process_time = round(end_time - start_time, 2)
        st.sidebar.metric(label="Vectorization Time (GPU)", value=f"{process_time} seconds")

        return vectorstore
    except Exception as e:
        st.sidebar.error(f"Error processing document: {e}")
        return None
    finally:
        # 3. Clean up the temporary file
        os.unlink(tmp_file_path)


def run_rag_query(query, vectorstore):
    """Performs RAG query (Retrieval + Generation)."""
    start_time = time.time()

    # 1. Retrieval
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = set([doc.metadata.get('page', 'Unknown') for doc in docs])
    source_refs = ", ".join([f"Page {s + 1}" for s in sources if isinstance(s, int)])

    # 2. Generation (LLM Call)
    try:
        llm = Ollama(model=LLM_MODEL)
        rag_prompt = PromptTemplate.from_template(
            "You are a helpful, professional assistant. Use the following context to provide a **comprehensive and detailed answer** to the question. Structure your response clearly using paragraphs or lists, referencing only the provided context. If the context does not contain the answer, state that you cannot answer based on the document. **Respond entirely in the language of the user's question.**\n\nContext: {context}\n\nQuestion: {query}"
        )
        chain = rag_prompt | llm
        response = chain.invoke({"context": context, "query": query})
    except Exception:
        # Simplified simulation to avoid errors with complex formatting
        response = f"**Simulated Response (Cannot Connect to Ollama):** Based on the retrieved context from pages {source_refs}, the model would have answered using the following information: {context[:500]}..."

    end_time = time.time()
    query_time = round(end_time - start_time, 2)
    return response, source_refs, query_time


def run_chat_query(prompt, chat_history):
    """Performs simple conversational query with memory."""
    start_time = time.time()

    # Structure history for the LLM prompt
    # Limit history to the last 5 turns to save VRAM and speed up processing
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-5:]])

    # Simple Chat Prompt
    chat_prompt = PromptTemplate.from_template(
        "You are a helpful and engaging general-purpose assistant named Gemini. Respond to the user's last question based on the full conversation history. **Respond entirely in the language of the user's question.**\n\nHistory: {history}\n\nUser Question: {prompt}"
    )

    try:
        llm = Ollama(model=LLM_MODEL)
        chain = chat_prompt | llm
        response = chain.invoke({"history": history_text, "prompt": prompt})
    except Exception:
        response = "**Simulated Chat Response (Cannot Connect to Ollama):** Hello! I'm your local AI assistant. I couldn't connect to the local model, but I'm ready to chat once the server is running."

    end_time = time.time()
    query_time = round(end_time - start_time, 2)
    return response, "", query_time


# --- Streamlit UI Setup ---

st.set_page_config(page_title="Document Q&A Bot", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Modern Dark Mode
st.markdown("""
    <style>
    :root {
        --bg-primary: #171717;
        --bg-secondary: #242424;
        --text-primary: #ECECEC;
        --accent-color: #4B98E9; /* Modern Blue */
        --border-color: #383838;
    }
    body, .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    .stTitle {
        background: -webkit-linear-gradient(90deg, var(--text-primary), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        border-bottom: 2px solid var(--border-color);
        padding: 10px 0;
        margin: 0;
    }
    /* Chat Area */
    .stChatMessage {
        background-color: var(--bg-secondary);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    /* Input Box */
    .stTextInput > div > div > input {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border-radius: 20px;
        border: 1px solid var(--border-color);
        padding: 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 10px rgba(75, 152, 233, 0.5);
    }
    /* Metrics */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    .stSidebar {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    </style>
""", unsafe_allow_html=True)

# --- Application Logic ---

st.title("🤖 Local AI Assistant")
st.caption("Dual-Mode: Conversational Chatbot or Private Document Q&A (RAG).")

# Sidebar Configuration and Mode Switch
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Mode Switch (Core UI Feature) ---
    MODE = st.radio("Select Application Mode:", ("Document Q&A (RAG)", "Local Chatbot"), index=0, key="mode")
    st.markdown("---")

    # Conditional UI for RAG Mode
    uploaded_file = None
    vector_db = None

    if MODE == "Document Q&A (RAG)":
        st.subheader("📚 RAG Setup")
        uploaded_file = st.file_uploader("Upload a PDF for Analysis", type="pdf")

    # Display performance metrics (populated inside run_query)
    st.subheader("📈 Performance Metrics")
    st.metric(label="Vectorization Time", value="N/A")
    st.metric(label="Query Time", value="N/A")

# --- Main Application Area ---

# Initialize messages and set dynamic welcome message
if "messages" not in st.session_state:
    st.session_state.messages = []

# Use a separate flag to ensure the welcome message only appears once when the mode changes
# This logic is what prevents chat history persistence, so we need to adjust it.
if "welcome_sent" not in st.session_state or st.session_state.get("last_mode") != MODE:
    # We must clear history only if the mode explicitly changed, not on every app run.
    if st.session_state.get("last_mode") != MODE:
        st.session_state.messages = []

        # Set welcome message only if history is empty
    if not st.session_state.messages:
        if MODE == "Document Q&A (RAG)":
            welcome_message = "Welcome! Upload a PDF in the sidebar to begin private Q&A."
        else:  # Local Chatbot
            welcome_message = "Hello! I'm your local AI assistant. Ask me anything."

        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

    st.session_state.welcome_sent = True
    st.session_state.last_mode = MODE  # Track the current mode

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input
if prompt := st.chat_input("What's your question?"):
    # Fix: User's prompt must be added to history before running the model
    user_query_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_query_message)

    # Display user's query immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine which logic to run based on the mode
    if MODE == "Document Q&A (RAG)" and uploaded_file:
        # 1. Load Document (will use cache if file is the same)
        vector_db = load_and_process_docs(uploaded_file)

        if vector_db:
            # --- RAG Mode Execution ---
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking... (RAG Search and Generation)"):
                    answer, sources, query_time = run_rag_query(prompt, vector_db)
                    full_response = f"**Answer:** {answer}\n\n**Source Pages:** {sources}"
                    st.markdown(full_response)

            # Update metrics and session state
            st.sidebar.metric(label="Vectorization Time", value=st.session_state.get("VectorizationTime", "Cached"))
            st.sidebar.metric(label="Query Time", value=f"{query_time} seconds")
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})  # Add assistant response to history

        else:
            # Handle document processing error
            with st.chat_message("assistant"):
                error_response = "⚠️ **Error:** Document processing failed. Please check the sidebar for details."
                st.markdown(error_response)
                st.session_state.messages.append({"role": "assistant", "content": error_response})


    elif MODE == "Local Chatbot":
        # --- Chatbot Mode Execution ---
        chat_history = st.session_state.messages
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking... (Conversational Generation)"):
                answer, _, query_time = run_chat_query(prompt, chat_history)
                st.markdown(answer)

        # Update metrics and session state
        st.sidebar.metric(label="Vectorization Time", value="N/A")
        st.sidebar.metric(label="Query Time", value=f"{query_time} seconds")
        st.session_state.messages.append({"role": "assistant", "content": answer})  # Add assistant response to history

    else:
        # Waiting state for RAG mode without file
        with st.chat_message("assistant"):
            waiting_message = "Please upload a document in the sidebar to start Q&A."
            st.markdown(waiting_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": waiting_message})  # Add waiting message to history
