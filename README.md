# 🍎 Mac-Optimized AI Assistant (RAG & Chatbot)

This Python application is a high-performance, dual-mode AI assistant built with **Streamlit** and **LangChain**. It is specifically optimized for Apple Silicon (M1, M2, M3) using **Metal Performance Shaders (MPS)** to accelerate vectorization and model inference via **Ollama**.

## ✨ Key Features

* **Dual-Mode Functionality:** Seamlessly switch between **Document Q&A (RAG)** and a **Local Conversational Chatbot**.
* **Apple Silicon Optimization:** Leverages the **Metal GPU (MPS)** for lightning-fast embeddings and model execution, significantly reducing latency compared to CPU-only solutions.
* **Built-in Caching:** Uses Streamlit's `@st.cache_resource` for efficient re-runs, minimizing document re-processing time.
* **Modern UI:** Features a sleek, custom **Dark Mode** user interface with Streamlit.
* **Multilingual Support:** The RAG prompt instructs the model to respond in the language of the user's question.

## 🛠️ Prerequisites

This application requires **Ollama** to be running locally with a specific Metal-accelerated model.

1.  **Install Ollama:**
    ```bash
    # Install the Ollama application from ollama.com
    ```
2.  **Install Dependencies:**
    ```bash
    pip install streamlit langchain-community langchain-core ollama pypdf faiss-cpu torch
    ```
    *Note: `torch` will automatically utilize MPS on Apple Silicon.*
3.  **Download Base Model:**
    ```bash
    ollama pull mistral
    ```
4.  **Create Metal-Accelerated Model:**
    The Python file expects a model named `mistral-metal-accel`. This requires a custom `Modelfile` to explicitly enable the GPU on Ollama.

    * Create a file named `Mistral-Metal.Modelfile`:
        ```
        FROM mistral
        PARAMETER num_gpu 99 # Tells Ollama to use the maximum available GPU layers (MPS)
        ```
    * Create the custom model (replace `mistral-metal-accel` if needed):
        ```bash
        ollama create mistral-metal-accel -f Mistral-Metal.Modelfile
        ```

## 🚀 How to Run

1.  **Ensure Ollama is running** in the background.
2.  **Execute the Streamlit application:**
    ```bash
    streamlit run your_file_name.py
    ```
    (Replace `your_file_name.py` with the actual file name).
3.  The application will open in your web browser.

## ⚙️ Configuration Notes

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | A fast, efficient model for vectorizing text. |
| `LLM_MODEL` | `mistral-metal-accel` | The locally running Ollama model, specifically configured for MPS. |
| **RAG Mode** | PDF Upload Required | Processes the uploaded PDF to create a FAISS vector database in-memory for private Q&A. |
| **Chatbot Mode** | No Upload Needed | Engages in a general conversation using the local LLM. |

***

_**Disclaimer:** The application assumes a correctly configured Ollama environment. If the model is not found or Ollama is not running, simulated responses will be displayed._
