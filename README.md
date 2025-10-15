🤖 Dual-Mode AI Assistant: M1-Optimized RAG and Local Chatbot
This project showcases a production-ready, highly efficient Dual-Mode AI Application built for private document analysis and general conversational support. It is specifically optimized to run locally on Apple Silicon (M1/M2/M3) devices, proving resourcefulness and MLOps capability on constrained hardware.

🌟 Technical Highlights (The Value Proposition)
This application is engineered to eliminate reliance on expensive cloud APIs and maximize local performance.

Feature

Technologies Demonstrated

MLOps Skill

End-to-End Metal Acceleration

PyTorch (MPS) & Ollama (Mistral)

Proven expertise in optimizing models for specialized, cost-effective hardware (M1 GPU).

Efficient Caching

@st.cache_resource & File Hashing

Builds responsive applications by preventing repetitive, costly document embedding (vectorization).

Persistent Memory

SQLite (Local Database)

Implementation of reliable, file-based persistence for chat history across sessions.

Dual-Mode Architecture

Streamlit Session State Management

Demonstrates ability to build complex applications with flexible functionality (RAG vs. Conversational).

Data Privacy

Local-Only LLM Inference

Guarantees sensitive client data never leaves the user's local machine.

⚠️ Critical Setup: Local LLM Server (Ollama)
This application relies entirely on the Ollama application and a local model. You must complete this setup for the application to function beyond the "Simulated Response."

1. Ollama Installation
Download: Install the Ollama application from their official website and launch it once to start the local server.

Model Download: Pull the base Mistral model to your Ollama library:

ollama pull mistral

2. GPU Acceleration Configuration
The rag_app.py is configured to use a GPU-optimized version named mistral-metal-accel. To build this accelerated version using your M1's GPU, follow these steps in your Mac Terminal:

Step A: Create the Custom Modelfile
Create a new file named Mistral-Metal.Modelfile and ensure it contains exactly these two lines:

FROM mistral
PARAMETER num_gpu 99

Step B: Build the Accelerated Model
Run the ollama create command to build the GPU-accelerated version:

ollama create mistral-metal-accel -f Mistral-Metal.Modelfile

Once this succeeds, the model is ready to run with maximum performance.

🗂️ Dependencies (requirements.txt)
You must install these packages in your Python Virtual Environment (venv):

streamlit
pandas
numpy
torch
transformers
sentence-transformers
pypdf
langchain-core
langchain-community
faiss-cpu
sqlite3

🚀 Quick Start
Setup: Install dependencies using pip install -r requirements.txt.

Run App:

streamlit run rag_app.py

Test Modes: Use the sidebar to switch between the fast, conversational "Local Chatbot" and the document-specific "Document Q&A (RAG)" modes.