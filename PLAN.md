# 📚 Local PDF RAG: Step-by-Step Execution Plan

This guide outlines how to run your local RAG (Retrieval-Augmented Generation) system on your machine using an open LLM (`llama3.2:1b`) and an embeddings model (`nomic-embed-text`) running on your GPU/CPU via **Ollama**.

---

## 🏗️ What is Already Done
We have initialized your workspace and prepared the core files:
1. **Virtual Environment**: Set up in `./venv` with `numpy`, `pypdf`, and `requests` pre-installed.
2. **Ollama Standalone Binary**: Downloaded to `./bin/ollama` and library dependencies placed in `./lib`.
3. **Ollama Startup Script** (`start_ollama.sh`): Sets up local models directories and runs the server.
4. **Core RAG Architecture** (`rag_system.py`):
   * **`PDFProcessor`**: Extracts text from PDFs and chunks it using a sliding window.
   * **`LocalVectorStore`**: Calls Ollama's local embeddings API, saves/loads vector indices as JSON files, and performs Cosine Similarity search via NumPy.
   * **`RAGPipeline`**: Assembles retrieved context, formats the system instructions/prompt, and queries the local LLM.
5. **Interactive CLI Client** (`query.py`): Starts a prompt interface to ask questions about your book.
6. **PDF Book**: Your book is located at `nossolar.pdf` in the workspace.

---

## 🚀 How to Run and Resume the Setup

Since all background processes in the sandbox environment terminate between tool runs, you can execute the remaining steps in your own terminal.

### Step 1: Activate the Python Virtual Environment
Open your terminal in the workspace folder (`/home/vhmolinar/dev/src/ai-stuff/rag`) and run:
```bash
source venv/bin/activate
```

### Step 2: Start the Local Ollama Server
Ollama runs in user space without needing root permissions. Start it using our script:
```bash
chmod +x start_ollama.sh
./start_ollama.sh
```
*Note: This script launches Ollama in the background (logs are redirected to `ollama.log`).*

### Step 3: Pull the Necessary Open Models
Download the embedding and generation models to Ollama. Since Ollama detected your **GTX 1660 Ti discrete GPU** in the logs, these models will run extremely fast:
```bash
export OLLAMA_HOST=127.0.0.1:11434

# 1. Pull the high-performance embedding model (~274MB)
./bin/ollama pull nomic-embed-text

# 2. Pull the lightweight instruction-following LLM (~1.3GB)
./bin/ollama pull llama3.2:1b
```

### Step 4: Run the Interactive RAG
Start the interactive program. It will read `nossolar.pdf`, split it into overlapping chunks, request vector embeddings from Ollama, and save the indexed database to `vector_index.json` (so it only indexes once!).
```bash
python3 query.py
```

### Step 5: Ask Questions!
Once the CLI is ready, you will see a prompt. You can ask questions about the book (e.g., *"What is Nosso Lar?"* or *"Who is André Luiz?"*).

---

## 🛠️ Key Architectural Details (Under the Hood)
* **Text Chunking**: We use a sliding window chunk size of 800 characters with a 150-character overlap. The chunker is smart enough to find the nearest word boundary so words aren't cut in half.
* **Vector Embeddings**: We call Ollama's `/api/embeddings` endpoint using `nomic-embed-text` which returns a 768-dimensional vector.
* **Vector DB**: Because Python 3.14 is highly experimental and compiled DB packages (like `chromadb` or `faiss`) can have build issues, we implemented a pure Python-and-NumPy vector search using **Cosine Similarity**:
  $$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
* **Prompt Injection**: The top $K$ relevant chunks are retrieved and formatted into a system-prompt constraint:
  *"If the answer cannot be found in the context, politely state that you do not know based on the provided material. Do not invent facts."*
