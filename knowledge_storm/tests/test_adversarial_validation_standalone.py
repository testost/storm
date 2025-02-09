"""Standalone integration tests for the Adversarial Validation System using production retrieval models."""

import os
import json
import pytest
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import requests
from knowledge_storm.rm import SerperRM

class TokenWiseRetriever:
    """Hybrid retrieval system with token-wise late interaction for better semantic matching."""
    
    def __init__(self):
        # Use multilingual model for both token and sentence embeddings
        self.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize SERPER client
        self.serper_key = os.getenv("SERPER_API_KEY")
        if not self.serper_key:
            raise ValueError("SERPER_API_KEY environment variable not set")
            
        self.rm = SerperRM(
            k=5,
            min_char_count=150,
            snippet_chunk_size=1000,
            webpage_helper_max_threads=10,
            ENABLE_EXTRA_SNIPPET_EXTRACTION=True,
            query_params={
                "num": 5,
                "autocorrect": True,
                "page": 1,
                "engine": "google",
                "type": "search"
            }
        )
    
    def _tokenwise_similarity(self, query: str, passage: str) -> float:
        """Compute token-wise similarity inspired by ColBERT's late interaction."""
        # Get token embeddings for query and passage
        query_tokens = query.split()
        passage_tokens = passage.split()
        
        # Encode each token (using mean pooling over wordpieces if needed)
        query_embeds = self.encoder.encode([t for t in query_tokens], convert_to_tensor=True)
        passage_embeds = self.encoder.encode([t for t in passage_tokens], convert_to_tensor=True)
        
        # Compute similarity matrix between all query and passage tokens
        sim_matrix = torch.matmul(query_embeds, passage_embeds.T)
        
        # MaxSim operation: for each query token, get max similarity to any passage token
        token_scores = torch.max(sim_matrix, dim=1)[0]
        
        # Final score is mean of token scores (could be weighted by IDF in future)
        return float(torch.mean(token_scores).item())
    
    def get_web_results(self, query: str, lang: str = "en", k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve initial results from web search using SERPER API."""
        try:
            headers = {
                'X-API-KEY': self.serper_key,
                'Content-Type': 'application/json'
            }
            payload = {
                'q': query,
                'gl': 'fi' if lang == 'fi' else 'us',
                'hl': lang,
                'num': k*2
            }
            response = requests.post(
                'https://google.serper.dev/search',
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract organic results
            results = []
            for item in data.get('organic', []):
                results.append({
                    'snippet': item.get('snippet', ''),
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'source': item.get('source', '')
                })
            return results
            
        except Exception as e:
            print(f"Web retrieval error: {e}")
            return []
    
    def dense_filter(self, query: str, results: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        """Filter results using dense retrieval."""
        if not results:
            return []
            
        query_embed = self.encoder.encode(query, convert_to_tensor=True)
        doc_embeds = self.encoder.encode(
            [res["snippet"] for res in results],
            convert_to_tensor=True
        )
        
        # Compute similarities
        scores = torch.matmul(query_embed.unsqueeze(0), doc_embeds.T).squeeze()
        top_indices = torch.argsort(scores, descending=True)[:k]
        
        return [results[idx.item()] for idx in top_indices]
    
    def rerank(self, query: str, results: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        """Rerank results using token-wise late interaction."""
        if not results:
            return []
            
        try:
            # Get token-wise similarity scores
            token_scores = [
                self._tokenwise_similarity(query, res["snippet"])
                for res in results
            ]
            
            # Create scored results
            scored_results = []
            for idx, score in enumerate(token_scores):
                result = results[idx].copy()
                result["score"] = float(score)
                result["token_score"] = float(score)
                result["rank"] = idx
                scored_results.append(result)
            
            # Sort by token-wise score
            return sorted(scored_results, key=lambda x: x["score"], reverse=True)[:k]
                
        except Exception as e:
            print(f"Reranking error: {e}")
            return results[:k]
    
    def retrieve(self, query: str, lang: str = "en", k: int = 5) -> List[Dict[str, Any]]:
        """Full retrieval pipeline: web search → dense filtering → reranking."""
        # Stage 1: Web retrieval
        web_results = self.get_web_results(query, lang, k=k*2)
        if not web_results:
            return []
            
        # Stage 2: Dense filtering
        filtered_results = self.dense_filter(query, web_results, k=k*2)
        if not filtered_results:
            return web_results[:k]
            
        # Stage 3: Reranking
        return self.rerank(query, filtered_results, k=k)


@pytest.fixture(scope="module")
def retriever():
    """Test fixture providing configured TokenWiseRetriever."""
    return TokenWiseRetriever()


def test_validate_english_article(retriever):
    """Test adversarial validation on English technical claims."""
    claim = "AI tools increase developer productivity by 40%"
    results = retriever.retrieve(claim, lang="en")
    
    # Verify we got results
    assert len(results) >= 3, "Should find at least 3 results"
    
    # Check result quality
    assert any(
        "productivity" in res["snippet"].lower() 
        for res in results
    ), "Should find productivity-related content"
    
    # Look for contradictions to the 40% claim
    assert any(
        "40%" not in res["snippet"] and any(
            term in res["snippet"].lower() 
            for term in ["increase", "productivity", "efficiency"]
        )
        for res in results
    ), "Should find alternative productivity metrics"
    
    print("\nEnglish Results:")
    for res in results:
        print(f"\nRank: {res.get('rank', 'N/A')}")
        print(f"Score: {res.get('score', 'N/A')}")
        print(f"Token Score: {res.get('token_score', 'N/A')}")
        print(f"Serper Score: {res.get('serper_score', 'N/A')}")
        print(f"Snippet: {res['snippet'][:200]}...")


def test_validate_finnish_article(retriever):
    """Test adversarial validation on Finnish music claims."""
    claim = "Suomalaiset metallibändit voittavat 80% pohjoismaisista palkinnoista"
    results = retriever.retrieve(claim, lang="fi")
    
    # Verify we got results
    assert len(results) >= 3, "Should find at least 3 results"
    
    # Check for Finnish content
    assert any(
        any(term in res["snippet"].lower() for term in ["suomi", "finland"])
        for res in results
    ), "Should find Finland-related content"
    
    # Look for music award statistics
    assert any(
        "80%" not in res["snippet"] and any(
            term in res["snippet"].lower() 
            for term in ["palkinto", "voitto", "award", "win"]
        )
        for res in results
    ), "Should find alternative award statistics"
    
    print("\nFinnish Results:")
    for res in results:
        print(f"\nRank: {res.get('rank', 'N/A')}")
        print(f"Score: {res.get('score', 'N/A')}")
        print(f"Token Score: {res.get('token_score', 'N/A')}")
        print(f"Serper Score: {res.get('serper_score', 'N/A')}")
        print(f"Snippet: {res['snippet'][:200]}...")


def test_multilingual_handling(retriever):
    """Test language-specific retrieval and source prioritization."""
    en_results = retriever.retrieve("GPT-4 coding benchmarks", lang="en")
    fi_results = retriever.retrieve("Älyteknologia kehitys Suomessa", lang="fi")
    
    # Verify English results
    assert any(
        "study" in res["source"].lower() or "research" in res["source"].lower()
        for res in en_results
    ), "Should find academic/research sources for English query"
    
    # Verify Finnish results
    assert any(
        ".fi" in res.get("link", "") or "suomi" in res.get("source", "").lower()
        for res in fi_results
    ), "Should find Finnish sources for Finnish query"
    
    print("\nMultilingual Test Results:")
    print("\nEnglish Query Results:")
    for res in en_results[:2]:
        print(f"Source: {res.get('source', 'N/A')}")
        print(f"Link: {res.get('link', 'N/A')}")
        print(f"Token Score: {res.get('token_score', 'N/A')}")
        print(f"Serper Score: {res.get('serper_score', 'N/A')}")
    
    print("\nFinnish Query Results:")
    for res in fi_results[:2]:
        print(f"Source: {res.get('source', 'N/A')}")
        print(f"Link: {res.get('link', 'N/A')}")
        print(f"Token Score: {res.get('token_score', 'N/A')}")
        print(f"Serper Score: {res.get('serper_score', 'N/A')}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
