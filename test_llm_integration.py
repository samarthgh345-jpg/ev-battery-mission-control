import os
import sys
import time
from unittest.mock import patch, MagicMock

from src.rag_pipeline import RAGPipeline, get_llm_provider, OllamaProvider, OpenAIProvider

def run_tests():
    print("Initializing RAG Pipeline for Phase 10 Verification...")
    rag = RAGPipeline()
    if not rag.index:
        rag.build_index()
        
    print("\n=============================================")
    print("TEST 1: OLLAMA PROVIDER (Simulated Success)")
    print("=============================================")
    with patch('urllib.request.urlopen') as mock_urlopen:
        # Mock successful Ollama response
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"response": "BTMS regulates battery temperature to ensure safety and efficiency."}'
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        provider = get_llm_provider()
        assert isinstance(provider, OllamaProvider), "Expected OllamaProvider"
        
        res = rag.generate_answer("What does BTMS do?")
        print(f"Provider: {res['provider']}")
        print(f"Answer: {res['answer']}")
        assert res['provider'] == "OllamaProvider"

    print("\n=============================================")
    print("TEST 2: OPENAI FALLBACK (Simulated Ollama Down)")
    print("=============================================")
    # Set fake API key
    os.environ["OPENAI_API_KEY"] = "fake-key-123"
    
    with patch('urllib.request.urlopen') as mock_urlopen:
        # First call is to Ollama "test" (fails), second is to OpenAI (succeeds)
        def side_effect(req, *args, **kwargs):
            if hasattr(req, 'full_url') and "localhost:11434" in req.full_url:
                raise Exception("Connection Refused")
            elif isinstance(req, str) and "localhost:11434" in req:
                raise Exception("Connection Refused")
            else:
                mock_response = MagicMock()
                mock_response.read.return_value = b'{"choices": [{"message": {"content": "OpenAI generated answer based on documents."}}]}'
                mock_response.__enter__.return_value = mock_response
                return mock_response
                
        mock_urlopen.side_effect = side_effect
        
        provider = get_llm_provider()
        assert isinstance(provider, OpenAIProvider), "Expected OpenAIProvider"
        
        res = rag.generate_answer("What is thermal runaway?")
        print(f"Provider: {res['provider']}")
        print(f"Answer: {res['answer']}")
        assert res['provider'] == "OpenAIProvider"

    print("\n=============================================")
    print("TEST 3: NO LLM (Fallback Mode)")
    print("=============================================")
    del os.environ["OPENAI_API_KEY"]
    
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_urlopen.side_effect = Exception("Connection Refused")
        
        provider = get_llm_provider()
        assert provider is None, "Expected None"
        
        res = rag.generate_answer("Why is liquid cooling used?")
        print(f"Provider: {res['provider']}")
        assert res['provider'] == "Fallback (No LLM)"
        assert "LLM unavailable" in res['answer']
        
    print("\n=============================================")
    print("TEST 4: FAILURE CASES")
    print("=============================================")
    
    # 1. Empty Query
    res_empty = rag.generate_answer("")
    # Empty string might just match nothing or everything depending on FAISS, 
    # but the UI prevents empty queries. Let's see what RAG returns.
    print(f"Empty query provider: {res_empty['provider']}")
    
    # 2. Out of domain
    res_ood = rag.generate_answer("What is the recipe for chocolate cake?")
    # Since FAISS IndexFlatL2 will still return nearest neighbors even if far, 
    # it might retrieve BTMS docs with high L2 distance.
    print(f"Out of domain top retrieved distance: {res_ood['sources'][0]['score'] if res_ood['sources'] else 'None'}")
    
    print("\nPhase 10 LLM Integration Verification Complete.")

if __name__ == "__main__":
    run_tests()
