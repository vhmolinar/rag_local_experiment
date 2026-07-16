# Local PDF RAG: Local Experiment & CLI Client

A 100% local, self-hosted Retrieval-Augmented Generation (RAG) system running on an open LLM (`llama3.2:1b`) and embedding model (`nomic-embed-text`) via **Ollama**.

This project indexes a local PDF book and provides an interactive CLI to query its contents with strict compliance rules to prevent model hallucinations.

---

## Architecture & Technical Highlights

* **Text Chunking**: Paragraph-aware chunking. Strips running headers/footers, joins hyphenated line wraps, and groups natural paragraphs up to ~800 characters with a 1-paragraph overlap.
* **Vector Store**: Pure Python and NumPy-based vector store using Cosine Similarity for search, saving/loading database indexes as plain `vector_index.json`.
* **Zero-Hallucination Guardrails**: Prompts are constrained using XML context tags, strict negative instructions (forcing "I do not know" fallback), and temperature set to `0.0` for deterministic generation.
* **Double Generation Pipelines**: Supports both a direct local Ollama generator and the **Google Antigravity TUI CLI (`agy`)** generator.

---

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3 and Ollama installed.

### 2. Install Dependencies
Set up the virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pypdf requests
```

### 3. Start the Ollama Server
Run the local Ollama daemon:
```bash
chmod +x start_ollama.sh
./start_ollama.sh
```

### 4. Pull the Models
Download the models to Ollama:
```bash
export OLLAMA_HOST=127.0.0.1:11434
./bin/ollama pull nomic-embed-text
./bin/ollama pull llama3.2:1b
```

---

## Running the CLI Clients

### Option A: Pure Local Ollama Pipeline
Run the interactive Python CLI. It will automatically build the `vector_index.json` database upon its first run:
```bash
python3 query.py
```

### Option B: Google Antigravity CLI Pipeline
If you have the Antigravity local CLI installed, you can use it for response generation:
```bash
python3 query_antigravity.py
```

---

## Project Structure

* [rag_system.py](file:///data/dev/src/ai-stuff/rag/rag_system.py): Core chunking, vector indexing, and pipeline logic.
* [query.py](file:///data/dev/src/ai-stuff/rag/query.py): Interactive prompt CLI using Ollama.
* [query_antigravity.py](file:///data/dev/src/ai-stuff/rag/query_antigravity.py): Interactive prompt CLI using the local `agy` command.
* [start_ollama.sh](file:///data/dev/src/ai-stuff/rag/start_ollama.sh): Script to launch the local Ollama server.
* [PLAN.md](file:///data/dev/src/ai-stuff/rag/PLAN.md): Step-by-step roadmap of the project.
