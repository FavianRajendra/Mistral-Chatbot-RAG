🤖 Local AI Assistant: Dual-Mode RAG & Chatbot (M-series Optimized)

This project is a high-performance, dual-mode AI application built with Streamlit, LangChain, and Ollama, specifically optimized for Apple Silicon (M1/M2/M3) Macs. It offers both a private Document Q&A (RAG) mode and a general-purpose Local Chatbot mode.

It is designed to showcase the power of on-device processing by utilizing Metal Performance Shaders (MPS) for hardware acceleration on both the vectorization and LLM generation steps.

✨ Key Features

Apple Silicon Optimization (MPS): Explicitly targets the Metal GPU for faster embedding (vectorization) and LLM inference, dramatically reducing query latency.

Dual Mode Functionality: Seamlessly switch between Document Q&A (RAG) for private file analysis and a general Local Chatbot.

Caching for Performance: Uses @st.cache_resource to ensure documents are only processed once per session, even with application reruns.

Modern Dark Mode UI: Custom CSS provides a clean, professional, and eye-friendly dark mode interface.

Multilingual Ready: Prompts are structured to ensure the model responds entirely in the language of the user's question.

Clear Metrics: Real-time feedback on vectorization and query times are displayed in the sidebar.

🛠️ Prerequisites

To run this application, you must have the following installed:

Ollama: The easiest way to run local LLMs with Metal acceleration.

Mistral LLM (Optimized): A specific version of the Mistral model created to maximize Metal acceleration.

Python Environment: Python 3.9+ is recommended.

1. Ollama Setup

First, download and install Ollama for macOS.

Next, you need to pull the base model and then create the optimized version used in the code (mistral-metal-accel).

Pull the base Mistral model:

ollama pull mistral


Create a Mistral-Metal.Modelfile (This file tells Ollama to load the model with specific optimizations):

FROM mistral
PARAMETER num_gpu 999


Note: num_gpu 999 forces Ollama to use the maximum available GPU layers, ensuring Metal acceleration is fully utilized.

Create the optimized model in Ollama:

ollama create mistral-metal-accel -f Mistral-Metal.Modelfile


This creates the specific model named mistral-metal-accel that the Streamlit app is configured to use.

2. Python Dependencies

Install the necessary Python packages.

# It is highly recommended to use a virtual environment
pip install streamlit langchain-community langchain-core pypdf transformers faiss-cpu torch


🚀 How to Run the Application

Save the Code: Save the Python code above as app.py.

Start Ollama: Ensure the Ollama server is running in the background.

Start the Streamlit App:

streamlit run app.py


The application will open automatically in your browser.

⚙️ Core Technical Implementation Details

Metal Acceleration (MPS)

The code explicitly checks for and utilizes the PyTorch MPS backend for hardware acceleration:

# In get_embedding_function()
if torch.backends.mps.is_available():
    DEVICE = "mps"
    # ...
model_kwargs = {'device': DEVICE}
return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)


RAG Flow

Loading: User uploads a PDF.

Caching: The load_and_process_docs function, decorated with @st.cache_resource, ensures the expensive vectorization step is only run once per uploaded file. The unique file_id acts as the cache key.

Embedding: Documents are split using RecursiveCharacterTextSplitter and embedded using the MPS-accelerated HuggingFaceEmbeddings (using all-MiniLM-L6-v2).

Retrieval: FAISS.similarity_search finds the most relevant document chunks.

Generation: The relevant context and user question are passed to the locally running Ollama Mistral model via a tailored PromptTemplate.
