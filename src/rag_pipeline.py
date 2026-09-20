"""
RAG Pipeline
============
Retrieval-Augmented Generation for Engineering Knowledge Assistance.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

# ── LLM Provider Abstraction ────────────────────────────────

class LLMProvider:
    def generate(self, prompt: str) -> Optional[str]:
        raise NotImplementedError

class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        import urllib.request
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> Optional[str]:
        import urllib.request
        import json
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        req = urllib.request.Request(self.url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response")
        except Exception as e:
            return None

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def generate(self, prompt: str) -> Optional[str]:
        try:
            import urllib.request
            import json
            url = "https://api.openai.com/v1/chat/completions"
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            return None

def get_llm_provider() -> Optional[LLMProvider]:
    """Retrieve an available LLM provider based on environment, else None."""
    # Primary option: OpenAI (if key is set)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return OpenAIProvider(openai_key)
        
    # Optional option: Try Ollama (assuming it might be running locally)
    ollama = OllamaProvider()
    if ollama.generate("test"):
        return ollama
        
    return None

# ── RAG System ──────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_dir: str = "models/rag"):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.json"
        
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        
        if self.index_path.exists() and self.metadata_path.exists():
            self.load_index()
            
    def _chunk_text(self, text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Simple word-based chunking."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "source": source,
                "chunk_id": f"{source}_{i}",
                "text": chunk_text
            })
        return chunks

    def build_index(self, knowledge_dir: str = "data/knowledge"):
        """Read documents, chunk, embed, and build FAISS index."""
        knowledge_path = Path(knowledge_dir)
        if not knowledge_path.exists():
            raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")
            
        chunks = []
        for file_path in knowledge_path.glob("*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                file_chunks = self._chunk_text(content, source=file_path.name)
                chunks.extend(file_chunks)
                
        if not chunks:
            print("No documents found to index.")
            return
            
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.metadata = chunks
        
        # Save
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
            
        print(f"Built FAISS index with {len(chunks)} chunks.")

    def load_index(self):
        """Load FAISS index and metadata."""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        if not self.index:
            raise ValueError("Index not built or loaded.")
            
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                dist = float(distances[0][i])
                if dist < 1.3:  # Relevance threshold
                    chunk = self.metadata[idx].copy()
                    chunk["score"] = dist
                    results.append(chunk)
                
        return results
        
    def generate_answer(self, query: str) -> Dict[str, Any]:
        """Generate a grounded answer using RAG."""
        retrieved_chunks = self.retrieve(query)
        
        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "sources": [],
                "provider": "None"
            }
            
        context_parts = []
        sources = set()
        for chunk in retrieved_chunks:
            context_parts.append(f"[{chunk['source']}]: {chunk['text']}")
            sources.add(chunk['source'])
            
        context_str = "\\n\\n".join(context_parts)
        
        prompt = f'''You are an EV battery thermal-management assistant.

Answer the user's question using ONLY the supplied retrieved context.
If the context does not contain enough information, say that the knowledge base does not provide enough information.
Do not invent numerical values.
Do not provide instructions for controlling real battery hardware.

Retrieved Context:
{context_str}

Question:
{query}'''

        provider = get_llm_provider()
        
        if provider:
            answer = provider.generate(prompt)
            if answer:
                return {
                    "answer": answer,
                    "sources": retrieved_chunks,
                    "provider": provider.__class__.__name__
                }
                
        # Fallback Mode when LLM is unavailable
        fallback_answer = (
            "⚠️ **LLM Configuration Required**\n\n"
            "The system successfully retrieved relevant context from the knowledge base, but no LLM provider is configured or available to generate a conversational answer.\n\n"
            "To enable conversational RAG, please set the `OPENAI_API_KEY` environment variable or ensure a local Ollama instance is running.\n\n"
            "**Retrieved Context Snippets:**\n\n"
        )
        for chunk in retrieved_chunks:
            fallback_answer += f"- **{chunk['source']}**: {chunk['text'][:200]}...\n"
            
        fallback_answer += "\n*The answer above is retrieval-based and has not been generated by an external language model.*"
        
        return {
            "answer": fallback_answer,
            "sources": retrieved_chunks,
            "provider": "Fallback (No LLM)"
        }
