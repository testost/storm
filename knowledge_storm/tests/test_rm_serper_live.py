import os
import pytest
import logging
logging.basicConfig(level=logging.INFO)
from knowledge_storm.rm import SerperRM

@pytest.fixture
def serper_rm():
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        pytest.skip("SERPER_API_KEY environment variable not set")
    return SerperRM(k=3)

def test_serper_rm_live_single_query(serper_rm):
    """Test a single query with live Serper API."""
    query = "latest news about artificial intelligence 2024"
    
    # Get raw response from Serper
    search_params = serper_rm.query_params.copy()
    search_params["q"] = query
    raw_result = serper_rm.serper_runner(search_params)
    print(f"\nRaw Serper response: {raw_result}")
    
    # Now try the forward method
    results = serper_rm.forward(query)
    print(f"\nProcessed results: {results}")
    
    assert len(results) > 0
    for result in results:
        assert "title" in result
        assert "url" in result
        assert "snippets" in result
        assert len(result["snippets"]) > 0

def test_serper_rm_live_multiple_queries(serper_rm):
    """Test multiple queries with live Serper API."""
    queries = [
        "latest AI developments 2024",
        "machine learning breakthroughs 2024"
    ]
    results = serper_rm.forward(queries)
    assert len(results) > 0
    for result in results:
        assert "title" in result
        assert "url" in result
        assert "snippets" in result
        assert len(result["snippets"]) > 0

def test_serper_rm_live_empty_query(serper_rm):
    """Test empty query with live Serper API."""
    results = serper_rm.forward("")
    assert len(results) == 0

def test_serper_rm_live_none_query(serper_rm):
    """Test None query with live Serper API."""
    results = serper_rm.forward(None)
    assert len(results) == 0

def test_serper_rm_live_empty_list(serper_rm):
    """Test empty list query with live Serper API."""
    results = serper_rm.forward([])
    assert len(results) == 0

def test_serper_rm_live_list_with_empty_queries(serper_rm):
    """Test list with empty queries with live Serper API."""
    results = serper_rm.forward(["", None, "  "])
    assert len(results) == 0

def test_serper_rm_live_exclude_urls(serper_rm):
    """Test excluding URLs with live Serper API."""
    query = "latest news about artificial intelligence 2024"
    # First get some results
    results1 = serper_rm.forward(query)
    assert len(results1) > 0
    
    # Then exclude the first URL
    url_to_exclude = results1[0]["url"]
    results2 = serper_rm.forward(query, exclude_urls=[url_to_exclude])
    
    # Verify the excluded URL is not in the results
    for result in results2:
        assert result["url"] != url_to_exclude

def test_serper_rm_live_special_characters(serper_rm):
    """Test queries with special characters with live Serper API."""
    special_queries = [
        "test!@#$%^&*()",
        "test\nquery",
        "test\tquery",
        "test'query",
        'test"query'
    ]
    for query in special_queries:
        results = serper_rm.forward(query)
        # We don't assert length as some queries might not return results
        for result in results:
            assert "title" in result
            assert "url" in result
            assert "snippets" in result

def test_serper_rm_live_unicode(serper_rm):
    """Test queries with Unicode characters with live Serper API."""
    results = serper_rm.forward("artificial intelligence 人工智能")
    assert len(results) > 0
    for result in results:
        assert "title" in result
        assert "url" in result
        assert "snippets" in result
