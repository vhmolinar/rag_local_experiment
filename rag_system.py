import json
import os
import requests
import numpy as np
from pypdf import PdfReader

class PDFProcessor:
    """Handles loading, cleaning, and paragraph-based chunking of PDF documents."""
    
    @staticmethod
    def extract_text(pdf_path):
        """Extracts all text page-by-page from a PDF file, stripping headers/footers."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        print(f"Reading PDF: {pdf_path}...")
        reader = PdfReader(pdf_path)
        pages_paragraphs = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Clean page and split into paragraphs
                paragraphs = PDFProcessor._clean_and_split_paragraphs(text)
                if paragraphs:
                    pages_paragraphs.append((i + 1, paragraphs))
                
        print(f"Successfully extracted paragraphs from {len(pages_paragraphs)} pages.")
        return pages_paragraphs

    @staticmethod
    def _clean_and_split_paragraphs(text):
        """Removes running headers/footers and reconstructs natural paragraphs."""
        import re
        
        # 1. Split into lines to remove running header & page numbers
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue
            # Skip running header
            if "Francisco Cândido Xavier - Nosso Lar" in stripped:
                continue
            # Skip page numbers (just digits)
            if stripped.isdigit():
                continue
            cleaned_lines.append(line)
            
        # Reassemble cleaned lines
        cleaned_text = "\n".join(cleaned_lines)
        
        # 2. Reassemble hyphenated words at line breaks (e.g. apren-\ndizes -> aprendizes)
        cleaned_text = re.sub(r'-\s*\n\s*', '', cleaned_text)
        
        # 3. Mark paragraph breaks (. \n or ? \n or ! \n)
        cleaned_text = re.sub(r'([.?!])\s*\n', r'\1<PARA_BREAK>', cleaned_text)
        
        # Replace remaining newlines (which are just line wraps inside a paragraph) with spaces
        cleaned_text = cleaned_text.replace('\n', ' ')
        
        # Collapse multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Restore paragraph breaks
        cleaned_text = cleaned_text.replace('<PARA_BREAK>', '\n\n')
        
        # Split into individual paragraphs
        paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
        return paragraphs

    @staticmethod
    def chunk_text(pages_paragraphs, target_chunk_size=800, overlap_paragraphs=1):
        """Groups paragraphs into chunks of roughly target_chunk_size with paragraph overlap."""
        print(f"Grouping paragraphs into chunks (target_size={target_chunk_size}, overlap_paragraphs={overlap_paragraphs})...")
        
        # Flatten all paragraphs with their source page number
        flat_paragraphs = []
        for page_num, paragraphs in pages_paragraphs:
            for para in paragraphs:
                flat_paragraphs.append({
                    "text": para,
                    "page": page_num
                })
                
        chunks = []
        i = 0
        while i < len(flat_paragraphs):
            current_chunk_paras = []
            current_size = 0
            
            # Start adding paragraphs from index i
            j = i
            while j < len(flat_paragraphs):
                para = flat_paragraphs[j]
                # Allow paragraph if it's the first one, or if adding it stays within 1.5x of target size
                if current_size == 0 or current_size + len(para["text"]) <= target_chunk_size * 1.5:
                    current_chunk_paras.append(para)
                    current_size += len(para["text"]) + 2  # +2 for paragraph spacer (\n\n)
                    j += 1
                else:
                    break
                    
            # Build chunk text and metadata
            chunk_text = "\n\n".join([p["text"] for p in current_chunk_paras])
            pages = list(set([p["page"] for p in current_chunk_paras]))
            pages.sort()
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "page": pages[0],
                    "pages": pages
                }
            })
            
            # Step forward: consumed paragraphs minus overlap
            paragraphs_consumed = j - i
            step = max(1, paragraphs_consumed - overlap_paragraphs)
            i += step
            
        print(f"Generated {len(chunks)} paragraph-aware chunks.")
        return chunks


class LocalVectorStore:
    """Manages document embeddings and performs cosine similarity retrieval."""
    
    def __init__(self, ollama_url="http://127.0.0.1:11434", embedding_model="nomic-embed-text"):
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.index = []  # List of dicts: {"text": str, "metadata": dict, "embedding": list}

    def _get_embedding(self, text):
        """Calls the local Ollama API to get the embedding vector for a text string."""
        url = f"{self.ollama_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding from Ollama: {e}")

    def add_documents(self, chunks):
        """Generates embeddings for all text chunks and adds them to the index."""
        print(f"Generating embeddings for {len(chunks)} chunks using '{self.embedding_model}'...")
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk["text"])
            self.index.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": embedding
            })
            if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
                print(f"Embedded {i + 1}/{len(chunks)} chunks...")

    def save(self, file_path):
        """Saves the indexed chunks and embeddings to a local JSON file."""
        print(f"Saving vector database to {file_path}...")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)
        print("Vector database saved successfully.")

    def load(self, file_path):
        """Loads the indexed chunks and embeddings from a local JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Index file not found: {file_path}")
        print(f"Loading vector database from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.index = json.load(f)
        print(f"Loaded vector database containing {len(self.index)} embedded chunks.")

    def search(self, query, k=3):
        """Performs cosine similarity search to find the top k most relevant chunks."""
        if not self.index:
            print("Vector database is empty.")
            return []
            
        # Get query embedding
        query_vector = np.array(self._get_embedding(query))
        
        results = []
        for item in self.index:
            doc_vector = np.array(item["embedding"])
            
            # Cosine Similarity calculation
            dot_product = np.dot(query_vector, doc_vector)
            norm_q = np.linalg.norm(query_vector)
            norm_d = np.linalg.norm(doc_vector)
            
            if norm_q > 0 and norm_d > 0:
                similarity = dot_product / (norm_q * norm_d)
            else:
                similarity = 0.0
                
            results.append((similarity, item))
            
        # Sort by similarity in descending order
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k results
        return results[:k]


class RAGPipeline:
    """Manages the query retrieval and LLM response generation loop."""
    
    def __init__(self, vector_store, ollama_url="http://127.0.0.1:11434", llm_model="llama3.2:1b"):
        self.vector_store = vector_store
        self.ollama_url = ollama_url
        self.llm_model = llm_model

    def ask(self, query, k=4, stream=True):
        """Retrieves top k chunks and feeds them into the prompt for generation."""
        # 1. Retrieve relevant contexts
        retrieved = self.vector_store.search(query, k=k)
        
        if not retrieved:
            return "Nenhum contexto relevante foi encontrado no banco de dados.", []
            
        # 2. Build the context string
        context_parts = []
        for i, (score, doc) in enumerate(retrieved):
            page_info = f"Página {doc['metadata']['page']}"
            context_parts.append(f"--- Contexto {i+1} ({page_info}, semelhança: {score:.3f}) ---\n{doc['text']}")
            
        context_str = "\n\n".join(context_parts)
        
        # 3. Design prompt template
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
        
        prompt = f"""Use o contexto fornecido dentro das tags <contexto> para responder à pergunta no final.

<contexto>
{context_str}
</contexto>

Pergunta: {query}
Resposta:"""

        # 4. Generate response from LLM
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.llm_model,
            "system": system_instructions,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.0
            }
        }
        
        # Print retrieval sources for transparency
        print("\n" + "="*40)
        print("📚 FONTES RECUPERADAS:")
        for score, doc in retrieved:
            print(f"- Página {doc['metadata']['page']} (Semelhança: {score:.3f}): \"{doc['text'][:80]}...\"")
        print("="*40 + "\n PROMPT GERADO:\n" + prompt + "\n" + "="*40)            
        print("="*40 + "\n🤖 RESPOSTA:")

        try:
            response = requests.post(url, json=payload, stream=stream)
            response.raise_for_status()
            
            full_response = ""
            if stream:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        chunk = data.get('response', '')
                        full_response += chunk
                        print(chunk, end='', flush=True)
                print("\n")  # Newline after stream finishes
            else:
                full_response = response.json()["response"]
                print(full_response)
                
            return full_response, retrieved
            
        except Exception as e:
            print(f"Error querying Ollama generation API: {e}")
            return f"Error: {e}", retrieved
