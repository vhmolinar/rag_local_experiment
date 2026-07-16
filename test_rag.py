import os
from rag_system import PDFProcessor, LocalVectorStore, RAGPipeline

def main():
    pdf_path = "book.pdf"
    db_path = "vector_index.json"
    ollama_url = "http://127.0.0.1:11434"
    embed_model = "nomic-embed-text"
    llm_model = "llama3.2:1b"
    
    db = LocalVectorStore(ollama_url=ollama_url, embedding_model=embed_model)
    
    if os.path.exists(db_path):
        print(f"[+] Loading existing index from {db_path}...")
        db.load(db_path)
    else:
        print(f"[+] Indexing PDF file: {pdf_path}...")
        pages = PDFProcessor.extract_text(pdf_path)
        chunks = PDFProcessor.chunk_text(pages)
        db.add_documents(chunks)
        db.save(db_path)
        print("[+] Indexing complete!")
        
    pipeline = RAGPipeline(vector_store=db, ollama_url=ollama_url, llm_model=llm_model)
    
    questions = [
        "Who is André Luiz?",
        "What is Nosso Lar?",
        "Who is Lias?"
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n==========================================")
        print(f"❓ Question {i}: {q}")
        print(f"==========================================")
        pipeline.ask(q, k=2, stream=False)
        print("\n")

if __name__ == "__main__":
    main()
