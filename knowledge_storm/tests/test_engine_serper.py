import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from ..storm_wiki.engine import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs
from ..rm import SerperRM

# Mock the serpapi module
mock_serpapi = MagicMock()
mock_google_search = MagicMock()
mock_serpapi.GoogleSearch.return_value = mock_google_search
sys.modules['serpapi'] = mock_serpapi

@pytest.fixture
def mock_serpapi_response():
    return {
        "search_metadata": {
            "id": "test_id",
            "status": "Success",
            "json_endpoint": "https://serpapi.com/searches/test_id/json",
            "created_at": "2024-02-08 19:35:28 UTC",
            "processed_at": "2024-02-08 19:35:28 UTC",
            "google_url": "https://www.google.com/search?q=test+query",
            "raw_html_file": "https://serpapi.com/searches/test_id/raw",
            "total_time_taken": 0.5
        },
        "search_parameters": {
            "engine": "google",
            "q": "test query",
            "google_domain": "google.com",
            "device": "desktop"
        },
        "organic_results": [
            {
                "position": 1,
                "title": "Test Result 1",
                "link": "https://example.com/1",
                "snippet": "This is the first test result snippet",
                "displayed_link": "example.com/1"
            },
            {
                "position": 2,
                "title": "Test Result 2",
                "link": "https://example.com/2",
                "snippet": "This is the second test result snippet",
                "displayed_link": "example.com/2"
            }
        ],
        "knowledge_graph": {
            "title": "Test Knowledge",
            "description": "Test knowledge graph description"
        }
    }

@pytest.fixture
def mock_serpapi_client(mock_serpapi_response):
    client = MagicMock()
    client.get_dict.return_value = mock_serpapi_response
    client.params_dict = {}  # Add this to simulate the params_dict attribute
    return client

@pytest.fixture
def engine(mock_serpapi_client):
    # Ensure we have a mock API key for testing
    if "SERPER_API_KEY" not in os.environ:
        os.environ["SERPER_API_KEY"] = "test_key"
    
    args = STORMWikiRunnerArguments(
        output_dir="test_output",
        max_conv_turn=3,
        max_perspective=3,
        max_search_queries_per_turn=3,
        search_top_k=3,
        retrieve_top_k=3,
        max_thread_num=10
    )
    
    lm_configs = STORMWikiLMConfigs()
    lm_configs.conv_simulator_lm = "gpt-4"
    lm_configs.question_asker_lm = "gpt-4"
    lm_configs.outline_gen_lm = "gpt-4"
    lm_configs.article_gen_lm = "gpt-4"
    lm_configs.article_polish_lm = "gpt-4"
    
    # Create SerperRM with mocked client
    rm = SerperRM(
        k=3,
        min_char_count=150,
        snippet_chunk_size=1000,
        webpage_helper_max_threads=10,
        ENABLE_EXTRA_SNIPPET_EXTRACTION=True,
        query_params={
            "num": 3,
            "autocorrect": True,
            "page": 1,
            "engine": "google",
            "type": "search"
        }
    )
    rm.serper_client = mock_serpapi_client
    
    return STORMWikiRunner(args=args, lm_configs=lm_configs, rm=rm)

def test_engine_initializes_serper_rm(engine):
    """Test that the engine properly initializes SerperRM"""
    assert isinstance(engine.retriever.rm, SerperRM)
    assert engine.retriever.rm.k == 3
    assert engine.retriever.rm.ENABLE_EXTRA_SNIPPET_EXTRACTION == True
    assert engine.retriever.rm.query_params["num"] == 3
    assert engine.retriever.rm.query_params["autocorrect"] == True
    assert engine.retriever.rm.query_params["type"] == "search"

def test_engine_search_with_serpapi(engine, mock_serpapi_response):
    """Test that the engine can perform a search using SerperRM"""
    # Perform a search
    query = "test query"
    results = engine.retriever.rm.forward(query, exclude_urls=[])

    # Verify the results
    assert len(results) == 2
    assert results[0]["title"] == "Test Result 1"
    assert results[0]["url"] == "https://example.com/1"
    assert results[0]["snippets"] == ["This is the first test result snippet"]
    assert results[0]["description"] == "Test knowledge graph description"

def test_engine_search_with_multiple_queries(engine, mock_serpapi_response):
    """Test that the engine can handle multiple search queries"""
    # Perform searches with multiple queries
    queries = ["query1", "query2"]
    results = engine.retriever.rm.forward(queries, exclude_urls=[])

    # Verify we got results for both queries
    assert len(results) == 4  # 2 results per query * 2 queries
    assert all(isinstance(r, dict) for r in results)
    assert all("title" in r and "url" in r and "snippets" in r for r in results)

def test_engine_search_handles_errors(engine):
    """Test that the engine properly handles API errors"""
    # Mock an error
    engine.retriever.rm.serper_client.get_dict.side_effect = Exception("API Error")

    # Perform a search that should fail
    results = engine.retriever.rm.forward("test query", exclude_urls=[])

    # Verify we get an empty result list on error
    assert isinstance(results, list)
    assert len(results) == 0

def test_engine_search_respects_k_limit(mock_serpapi_response):
    """Test that the engine respects the k limit for results"""
    # Create SerperRM with k=2
    rm = SerperRM(
        k=2,
        min_char_count=150,
        snippet_chunk_size=1000,
        webpage_helper_max_threads=10,
        ENABLE_EXTRA_SNIPPET_EXTRACTION=True,
        query_params={
            "num": 2,  # Also set num=2 to match k
            "autocorrect": True,
            "page": 1,
            "engine": "google",
            "type": "search"
        }
    )

    # Mock the serpapi client
    client = MagicMock()
    client.get_dict.return_value = mock_serpapi_response
    client.params_dict = {}  # Add this to simulate the params_dict attribute
    rm.serper_client = client

    # Perform a search
    results = rm.forward("test query", exclude_urls=[])

    # Verify we only get k=2 results
    assert len(results) == 2
