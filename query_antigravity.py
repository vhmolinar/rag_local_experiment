import os
import sys
import subprocess
import argparse
from rag_system import LocalVectorStore, PDFProcessor

DEFAULT_PDF = "book.pdf"
DB_FILE = "vector_index.json"
EMBEDDING_MODEL = "nomic-embed-text"

class AntigravityCLIRAGPipeline:
    """RAG Pipeline that uses the local 'agy' CLI tool for prompt generation."""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def ask(self, query, k=4):
        # 1. Retrieve contexts
        retrieved = self.vector_store.search(query, k=k)
        if not retrieved:
            print("Nenhum contexto relevante foi encontrado no banco de dados.")
            return
            
        # 2. Build Portuguese context string
        context_parts = []
        for i, (score, doc) in enumerate(retrieved):
            page_info = f"Página {doc['metadata']['page']}"
            context_parts.append(f"--- Contexto {i+1} ({page_info}, semelhança: {score:.3f}) ---\n{doc['text']}")
        context_str = "\n\n".join(context_parts)
        
        system_instructions = (
            "Você é um assistente de compreensão de leitura estrito. Você deve responder à pergunta do usuário usando APENAS o texto fornecido dentro das tags <contexto>.\n"
            "REGRAS CRÍTICAS:\n"
            "1. Não use nenhum conhecimento pré-existente ou faça suposições fora do texto.\n"
            "2. Se o contexto não contiver a resposta, você DEVE responder exatamente: 'Não tenho informações sobre isso com base no texto fornecido.'\n"
            "3. Não invente fatos ou extrapole.\n\n"
            "Exemplo 1:\n"
            "<contexto>\nO hospital de Nosso Lar fica na zona de transição.\n</contexto>\n"
            "Pergunta: Quem descobriu o Brasil?\n"
            "Resposta: Não tenho informações sobre isso com base no texto fornecido."
        )
        
        prompt = f"""Instruções do Sistema:
{system_instructions}

Use o contexto fornecido dentro das tags <contexto> para responder à pergunta no final.

<contexto>
{context_str}
</contexto>

Pergunta: {query}
Resposta:"""

        # Print sources
        print("\n" + "="*40)
        print("📚 FONTES RECUPERADAS:")
        for score, doc in retrieved:
            print(f"- Página {doc['metadata']['page']} (Semelhança: {score:.3f}): \"{doc['text'][:80]}...\"")
        print("="*40 + "\n🤖 RESPOSTA:")

        # 3. Call local 'agy' CLI tool using subprocess and stream output
        try:
            # We pass the system instruction as part of the prompt structure
            process = subprocess.Popen(
                ['agy', '--print', prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Read and print character by character to stream output
            while True:
                char = process.stdout.read(1)
                if not char and process.poll() is not None:
                    break
                if char:
                    sys.stdout.write(char)
                    sys.stdout.flush()
            
            # Flush any remaining outputs
            stdout, stderr = process.communicate()
            if stdout:
                sys.stdout.write(stdout)
                sys.stdout.flush()
            print() # Newline
            
            if process.returncode != 0:
                print(f"\n[-] Erro do Antigravity CLI (Código {process.returncode}): {stderr}")
                
        except FileNotFoundError:
            print("\n[-] Erro: O executável 'agy' não foi encontrado no PATH.")
            print("    Certifique-se de que o Antigravity CLI está instalado e configurado.")
        except Exception as e:
            print(f"\n[-] Erro ao chamar Antigravity: {e}")

def main():
    parser = argparse.ArgumentParser(description="Antigravity CLI RAG System")
    parser.add_argument("--pdf", type=str, default=DEFAULT_PDF, help="Path to the PDF file to index")
    parser.add_argument("--db", type=str, default=DB_FILE, help="Path to save/load the vector database index")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing of the PDF file")
    
    args = parser.parse_args()
    
    # Check if Ollama is running (still needed for embeddings generation if indexing)
    ollama_url = "http://127.0.0.1:11434"
    if args.reindex or not os.path.exists(args.db):
        try:
            import requests
            requests.get(ollama_url)
        except Exception:
            print(f"[-] Erro: Servidor Ollama local não encontrado em {ollama_url}.")
            print("    O Ollama é necessário para gerar os embeddings durante a indexação.")
            print("    Por favor, inicie o Ollama (execute `./start_ollama.sh`) antes de indexar.")
            return

    db = LocalVectorStore(ollama_url=ollama_url, embedding_model=EMBEDDING_MODEL)
    
    if os.path.exists(args.db) and not args.reindex:
        db.load(args.db)
    else:
        if not os.path.exists(args.pdf):
            print(f"[-] Erro: Arquivo PDF não encontrado em {args.pdf}")
            return
        print(f"[+] Indexando PDF: {args.pdf}")
        pages = PDFProcessor.extract_text(args.pdf)
        chunks = PDFProcessor.chunk_text(pages)
        db.add_documents(chunks)
        db.save(args.db)
        print("[+] PDF indexado com sucesso!")
        
    pipeline = AntigravityCLIRAGPipeline(vector_store=db)
    
    print("\n" + "="*50)
    print("🚀 SISTEMA RAG ANTIGRAVITY (VIA CLI) PRONTO!")
    print(f"   Origem do PDF: {args.pdf}")
    print(f"   Embeddings:    {EMBEDDING_MODEL}")
    print(f"   Gerador:       Local 'agy' CLI Tool")
    print("="*50)
    print("Digite suas perguntas abaixo. Digite 'exit' ou 'quit' para sair.\n")
    
    while True:
        try:
            query = input("\n📝 Faça uma pergunta: ").strip()
            if not query:
                continue
            if query.lower() in ('exit', 'quit'):
                print("Até logo!")
                break
                
            pipeline.ask(query)
        except (KeyboardInterrupt, EOFError):
            print("\nAté logo!")
            break
        except Exception as e:
            print(f"\n[-] Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
