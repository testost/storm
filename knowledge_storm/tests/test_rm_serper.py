import os
import pytest
from unittest.mock import patch, MagicMock
from knowledge_storm.rm import SerperRM

@pytest.fixture
def serper_rm():
    os.environ["SERPER_API_KEY"] = "test_key"
    return SerperRM(k=3)

def test_serper_rm_initialization():
    # Test initialization with environment variable
    os.environ["SERPER_API_KEY"] = "test_key"
    rm = SerperRM()
    assert rm.serper_search_api_key == "test_key"
    assert rm.k == 3

    # Test initialization with direct key
    rm = SerperRM(serper_search_api_key="direct_key")
    assert rm.serper_search_api_key == "direct_key"

    # Test initialization without key
    os.environ.pop("SERPER_API_KEY", None)
    with pytest.raises(RuntimeError):
        SerperRM()

def test_serper_rm_forward_single_query(serper_rm):
    mock_result = {
        "organic_results": [
            {
                "title": "Test Title 1",
                "link": "http://test1.com",
                "snippet": "Test Snippet 1"
            },
            {
                "title": "Test Title 2",
                "link": "http://test2.com",
                "snippet": "Test Snippet 2"
            }
        ],
        "knowledge_graph": {
            "description": "Knowledge Graph Description"
        }
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        results = serper_rm.forward("test query")
        assert len(results) == 2
        assert results[0]["title"] == "Test Title 1"
        assert results[0]["url"] == "http://test1.com"
        assert results[0]["snippets"] == ["Test Snippet 1"]
        assert results[0]["description"] == "Knowledge Graph Description"

def test_serper_rm_forward_multiple_queries(serper_rm):
    mock_result = {
        "organic_results": [
            {
                "title": "Test Title",
                "link": "http://test.com",
                "snippet": "Test Snippet"
            }
        ],
        "knowledge_graph": {}
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        results = serper_rm.forward(["query1", "query2"])
        assert len(results) == 2  # One result for each query
        for result in results:
            assert result["title"] == "Test Title"
            assert result["url"] == "http://test.com"
            assert result["snippets"] == ["Test Snippet"]

def test_serper_rm_exclude_urls(serper_rm):
    mock_result = {
        "organic_results": [
            {
                "title": "Test Title 1",
                "link": "http://exclude.com",
                "snippet": "Test Snippet 1"
            },
            {
                "title": "Test Title 2",
                "link": "http://include.com",
                "snippet": "Test Snippet 2"
            }
        ],
        "knowledge_graph": {}
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        results = serper_rm.forward("test query", exclude_urls=["http://exclude.com"])
        assert len(results) == 1
        assert results[0]["url"] == "http://include.com"

def test_serper_rm_empty_results(serper_rm):
    mock_result = {
        "organic_results": [],
        "knowledge_graph": {}
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        results = serper_rm.forward("test query")
        assert len(results) == 0

def test_serper_rm_missing_fields(serper_rm):
    mock_result = {
        "organic_results": [
            {
                # Missing title
                "link": "http://test.com",
                "snippet": "Test Snippet"
            }
        ]
        # Missing knowledge_graph
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        results = serper_rm.forward("test query")
        assert len(results) == 1
        assert results[0]["title"] == ""  # Should handle missing title
        assert results[0]["description"] == "Test Snippet"  # Should fall back to snippet

def test_serper_rm_api_error(serper_rm):
    with patch.object(serper_rm.serper_client, 'get_dict', side_effect=Exception("API Error")):
        results = serper_rm.forward("test query")
        assert len(results) == 0  # Should return empty list on error

def test_serper_rm_malformed_response(serper_rm):
    mock_result = {
        # Missing organic_results key
        "some_other_key": []
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        results = serper_rm.forward("test query")
        assert len(results) == 0

def test_serper_rm_malformed_item(serper_rm):
    mock_result = {
        "organic_results": [
            {
                # Missing required fields
                "some_field": "some value"
            }
        ],
        "knowledge_graph": {}
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        results = serper_rm.forward("test query")
        assert len(results) == 1
        assert results[0]["title"] == ""
        assert results[0]["url"] == ""
        assert results[0]["snippets"] == [""]

def test_serper_rm_empty_query(serper_rm):
    results = serper_rm.forward("")
    assert len(results) == 0

def test_serper_rm_none_query(serper_rm):
    results = serper_rm.forward(None)
    assert len(results) == 0

def test_serper_rm_empty_list_query(serper_rm):
    results = serper_rm.forward([])
    assert len(results) == 0

def test_serper_rm_list_with_empty_queries(serper_rm):
    results = serper_rm.forward(["", None, "  "])
    assert len(results) == 0

def test_serper_rm_duplicate_queries(serper_rm):
    mock_result = {
        "organic_results": [
            {
                "title": "Test Title",
                "link": "http://test.com",
                "snippet": "Test Snippet"
            }
        ],
        "knowledge_graph": {}
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        # Test with duplicate queries
        results = serper_rm.forward(["query", "query", "query"])
        assert len(results) == 3  # Should still process each query separately

def test_serper_rm_special_characters(serper_rm):
    mock_result = {
        "organic_results": [
            {
                "title": "Test Title",
                "link": "http://test.com",
                "snippet": "Test Snippet"
            }
        ],
        "knowledge_graph": {}
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        # Test with queries containing special characters
        special_queries = [
            "test!@#$%^&*()",
            "test\nquery",
            "test\tquery",
            "test'query",
            'test"query'
        ]
        for query in special_queries:
            results = serper_rm.forward(query)
            assert len(results) == 1

def test_serper_rm_large_response(serper_rm):
    # Create a large response with many results
    mock_result = {
        "organic_results": [
            {
                "title": f"Title {i}",
                "link": f"http://test{i}.com",
                "snippet": f"Snippet {i}"
            }
            for i in range(100)  # Large number of results
        ],
        "knowledge_graph": {
            "description": "Test Description"
        }
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        results = serper_rm.forward("test query")
        assert len(results) == serper_rm.k  # Should only return k results

def test_serper_rm_unicode_handling(serper_rm):
    mock_result = {
        "organic_results": [
            {
                "title": "Test 你好 Title",
                "link": "http://test.com",
                "snippet": "Test 你好 Snippet"
            }
        ],
        "knowledge_graph": {
            "description": "Test 你好 Description"
        }
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        results = serper_rm.forward("test 你好 query")
        assert len(results) == 1
        assert "你好" in results[0]["title"]
        assert "你好" in results[0]["snippets"][0]
        assert "你好" in results[0]["description"]

def test_serper_rm_concurrent_queries(serper_rm):
    mock_result = {
        "organic_results": [
            {
                "title": "Test Title",
                "link": "http://test.com",
                "snippet": "Test Snippet"
            }
        ],
        "knowledge_graph": {}
    }

    with patch.object(serper_rm.serper_client, 'get_dict', return_value=mock_result):
        # Test with a large number of concurrent queries
        large_query_list = [f"query{i}" for i in range(50)]
        results = serper_rm.forward(large_query_list)
        assert len(results) == 50  # Should handle all queries
