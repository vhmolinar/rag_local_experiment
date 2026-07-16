import os
import argparse
from rag_system import PDFProcessor, LocalVectorStore, RAGPipeline

DEFAULT_PDF = "nossolar.pdf"
DB_FILE = "vector_index.json"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"

def main():
    parser = argparse.ArgumentParser(description="Local RAG System CLI")
    parser.add_argument("--pdf", type=str, default=DEFAULT_PDF, help="Path to the PDF file to index")
    parser.add_argument("--db", type=str, default=DB_FILE, help="Path to save/load the vector database index")
    parser.add_argument("--model", type=str, default=LLM_MODEL, help="Ollama LLM generator model name")
    parser.add_argument("--embed-model", type=str, default=EMBEDDING_MODEL, help="Ollama embedding model name")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing of the PDF file")
    
    args = parser.parse_args()
    
    # Check if Ollama is running
    ollama_url = "http://127.0.0.1:11434"
    try:
        import requests
        requests.get(ollama_url)
    except Exception:
        print(f"[-] Error: Could not connect to local Ollama server at {ollama_url}.")
        print("    Please start the Ollama server first (run `./start_ollama.sh`).")
        return
        
    db = LocalVectorStore(ollama_url=ollama_url, embedding_model=args.embed_model)
    
    # Load or create index
    if os.path.exists(args.db) and not args.reindex:
        print(f"[+] Found existing vector database index: {args.db}")
        try:
            db.load(args.db)
        except Exception as e:
            print(f"[-] Error loading database: {e}. Reindexing...")
            index_needed = True
        else:
            index_needed = False
    else:
        index_needed = True
        
    if index_needed:
        if not os.path.exists(args.pdf):
            print(f"[-] Error: PDF file not found at {args.pdf}")
            print(f"    Please place a PDF file at {args.pdf} or specify --pdf <path>")
            return
            
        print(f"[+] Indexing new PDF: {args.pdf}")
        try:
            # Process PDF
            pages = PDFProcessor.extract_text(args.pdf)
            chunks = PDFProcessor.chunk_text(pages)
            
            # Add to vector database
            db.add_documents(chunks)
            db.save(args.db)
            print(f"[+] PDF indexing complete. Saved to {args.db}")
        except Exception as e:
            print(f"[-] Failed to index PDF: {e}")
            return
        
    # Setup pipeline
    pipeline = RAGPipeline(vector_store=db, ollama_url=ollama_url, llm_model=args.model)
    
    print("\n" + "="*50)
    print("🚀 LOCAL RAG SYSTEM READY!")
    print(f"   PDF Source:     {args.pdf}")
    print(f"   Embeddings:     {args.embed_model}")
    print(f"   Generator LLM:  {args.model}")
    print("="*50)
    print("Type your questions below. Enter 'exit' or 'quit' to end.\n")
    
    while True:
        try:
            query = input("\n📝 Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
                
            pipeline.ask(query, k=2)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[-] An error occurred: {e}")

if __name__ == "__main__":
    main()
