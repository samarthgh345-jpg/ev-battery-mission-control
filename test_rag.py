import os
from src.rag_pipeline import RAGPipeline

def main():
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()
    
    print("Building FAISS index from knowledge base...")
    rag.build_index()
    
    questions = [
        "What is battery thermal management?",
        "Why is cooling important in lithium-ion batteries?",
        "What is thermal runaway?",
        "What are the advantages of liquid cooling?",
        "What is the capital of France?" # Out of domain
    ]
    
    # Temporarily remove API keys to test fallback mode
    old_openai = os.environ.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
        
    print("\n--- Testing RAG (Fallback / Local Mode) ---")
    
    for q in questions:
        print(f"\nQuery: {q}")
        res = rag.generate_answer(q)
        print(f"Provider: {res['provider']}")
        print(f"Answer:\n{res['answer']}")
        print("\nRetrieved Sources:")
        for source in res['sources']:
            print(f"  - [{source['source']}] (Score: {source['score']:.4f})")
            
if __name__ == "__main__":
    main()
