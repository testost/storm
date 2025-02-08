import pytest
import requests_mock
from ..rm import YouRM
import time

@pytest.fixture
def you_rm():
    return YouRM(ydc_api_key="test_key", k=3)

def test_successful_search(you_rm):
    mock_response = {
        "hits": [
            {
                "url": "https://example.com/1",
                "title": "Test Result 1",
                "description": "Description 1",
                "snippets": ["Snippet 1"]
            },
            {
                "url": "https://example.com/2",
                "title": "Test Result 2",
                "description": "Description 2",
                "snippets": ["Snippet 2"]
            }
        ]
    }
    
    with requests_mock.Mocker() as m:
        m.get("https://api.ydc-index.io/search", json=mock_response)
        results = you_rm.forward("test query")
        
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["title"] == "Test Result 1"
        assert results[0]["description"] == "Description 1"
        assert results[0]["snippets"] == ["Snippet 1"]

def test_no_hits_in_response(you_rm):
    mock_response = {}  # Response without hits
    
    with requests_mock.Mocker() as m:
        m.get("https://api.ydc-index.io/search", json=mock_response)
        results = you_rm.forward("test query")
        
        assert len(results) == 0

def test_api_error_response(you_rm):
    with requests_mock.Mocker() as m:
        m.get("https://api.ydc-index.io/search", status_code=500)
        results = you_rm.forward("test query")
        
        assert len(results) == 0

def test_multiple_queries(you_rm):
    mock_response1 = {
        "hits": [
            {
                "url": "https://example.com/1",
                "title": "Test Result 1",
                "description": "Description 1",
                "snippets": ["Snippet 1"]
            }
        ]
    }
    
    mock_response2 = {
        "hits": [
            {
                "url": "https://example.com/2",
                "title": "Test Result 2",
                "description": "Description 2",
                "snippets": ["Snippet 2"]
            }
        ]
    }
    
    with requests_mock.Mocker() as m:
        m.get("https://api.ydc-index.io/search", [
            {'json': mock_response1},
            {'json': mock_response2}
        ])
        
        results = you_rm.forward(["query1", "query2"])
        
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/1"
        assert results[1]["url"] == "https://example.com/2"

def test_exclude_urls(you_rm):
    mock_response = {
        "hits": [
            {
                "url": "https://example.com/1",
                "title": "Test Result 1",
                "description": "Description 1",
                "snippets": ["Snippet 1"]
            },
            {
                "url": "https://example.com/2",
                "title": "Test Result 2",
                "description": "Description 2",
                "snippets": ["Snippet 2"]
            }
        ]
    }
    
    with requests_mock.Mocker() as m:
        m.get("https://api.ydc-index.io/search", json=mock_response)
        results = you_rm.forward("test query", exclude_urls=["https://example.com/1"])
        
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/2"

def test_respects_k_limit(you_rm):
    mock_response = {
        "hits": [
            {"url": f"https://example.com/{i}", 
             "title": f"Test Result {i}",
             "description": f"Description {i}",
             "snippets": [f"Snippet {i}"]} for i in range(5)
        ]
    }
    
    with requests_mock.Mocker() as m:
        m.get("https://api.ydc-index.io/search", json=mock_response)
        results = you_rm.forward("test query")
        
        # YouRM was initialized with k=3
        assert len(results) == 3

def test_rate_limit_handling(you_rm):
    mock_rate_limit_response = {
        'error_code': 'Too many requests',
        'message': 'You have hit your subscription limit!'
    }
    
    with requests_mock.Mocker() as m:
        m.get("https://api.ydc-index.io/search", json=mock_rate_limit_response)
        
        # First request should return empty list and set rate limit
        results = you_rm.forward("test query")
        assert len(results) == 0
        
        # Second request should skip due to rate limit window
        results = you_rm.forward("test query")
        assert len(results) == 0

def test_rate_limit_recovery(you_rm, monkeypatch):
    mock_rate_limit_response = {
        'error_code': 'Too many requests',
        'message': 'You have hit your subscription limit!'
    }
    
    mock_success_response = {
        "hits": [
            {
                "url": "https://example.com/1",
                "title": "Test Result 1",
                "description": "Description 1",
                "snippets": ["Snippet 1"]
            }
        ]
    }
    
    # Mock time.time() to control rate limit window
    current_time = [0]
    def mock_time():
        return current_time[0]
    monkeypatch.setattr(time, 'time', mock_time)
    
    with requests_mock.Mocker() as m:
        # First request hits rate limit
        m.get("https://api.ydc-index.io/search", json=mock_rate_limit_response)
        results = you_rm.forward("test query")
        assert len(results) == 0
        
        # Move time forward past rate limit window
        current_time[0] = 61  # 61 seconds later
        
        # Set up success response
        m.get("https://api.ydc-index.io/search", json=mock_success_response)
        results = you_rm.forward("test query")
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/1"
