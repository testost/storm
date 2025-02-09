import pytest
from typing import List
import dspy
import os
from knowledge_storm.storm_wiki.modules.counter_research import CounterResearcher

@pytest.fixture
def simple_claim():
    """Sample claim text for testing counter-evidence research."""
    return {
        "claim": "GPT-4 consistently outperforms GPT-3.5 in coding tasks",
        "confidence": 0.9,
        "context": "Model Performance",
        "source_text": "Studies have shown that GPT-4 consistently outperforms GPT-3.5 in coding tasks across multiple benchmarks."
    }

@pytest.fixture
def complex_claim():
    """Complex claim with multiple aspects to research."""
    return {
        "claim": "Remote work increases productivity by 55% while reducing operational costs",
        "confidence": 0.85,
        "context": "Workplace Efficiency",
        "source_text": "Research indicates that remote work arrangements led to a 55% increase in productivity while significantly reducing office-related operational costs."
    }

class MockRetrieve(dspy.Module):
    """Mock retrieval module for testing."""
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        
    def forward(self, query):
        """Return mock search results."""
        # Return opposing viewpoints based on query
        if "GPT-4" in query:
            return [
                dspy.Prediction({
                    "text": "While GPT-4 shows improvements, studies indicate that its performance advantage over GPT-3.5 varies significantly by task type, with some coding benchmarks showing only marginal gains.",
                    "source": "https://example.com/llm-benchmarks"
                }),
                dspy.Prediction({
                    "text": "Cost-benefit analysis reveals that GPT-3.5's faster inference time may make it more practical for many coding applications despite lower accuracy.",
                    "source": "doi:10.1234/ai.2024.789"
                })
            ]
        elif "remote work" in query.lower():
            return [
                dspy.Prediction({
                    "text": "Recent studies challenge the 55% productivity claim, showing that remote work benefits vary greatly by industry and role, with some sectors reporting decreased efficiency.",
                    "source": "https://example.com/remote-work-study"
                }),
                dspy.Prediction({
                    "text": "While operational costs decrease, companies report increased IT infrastructure and security expenses in remote work setups.",
                    "source": "doi:10.1234/management.2024.456"
                })
            ]
        return []

@pytest.fixture
def dspy_settings():
    """Global DSPy configuration for tests."""
    if not dspy.settings.lm:
        dspy.settings.configure(lm=dspy.LM("gpt-4o-mini"))
    return dspy.settings.lm

@pytest.fixture
def counter_researcher(dspy_settings):
    """Test fixture providing configured CounterResearcher with mock retrieval."""
    researcher = CounterResearcher()
    researcher.retrieve = MockRetrieve()
    return researcher

def test_counter_researcher_finds_opposing_evidence(counter_researcher, simple_claim):
    """Test that CounterResearcher can find opposing evidence for a claim."""
    counter_evidence = counter_researcher.find_counter_evidence(simple_claim)
    
    # Verify counter evidence structure
    assert len(counter_evidence) > 0
    for evidence in counter_evidence:
        assert "text" in evidence
        assert "source" in evidence
        assert "confidence" in evidence
        assert "relevance" in evidence
        
        # Verify it contains counter-arguments
        assert any(phrase in evidence["text"].lower() for phrase in [
            "varies significantly",
            "marginal gains",
            "cost-benefit",
            "despite lower"
        ])

def test_counter_researcher_handles_complex_claims(counter_researcher, complex_claim):
    """Test that CounterResearcher can handle claims with multiple aspects."""
    counter_evidence = counter_researcher.find_counter_evidence(complex_claim)
    
    # Verify multiple pieces of evidence addressing different aspects
    assert len(counter_evidence) >= 2  # Should find multiple counter-points
    
    # Check if evidence addresses both productivity and cost claims
    has_productivity_counter = False
    has_cost_counter = False
    
    for evidence in counter_evidence:
        if "productivity" in evidence["text"].lower():
            has_productivity_counter = True
        if "cost" in evidence["text"].lower() or "expense" in evidence["text"].lower():
            has_cost_counter = True
            
    assert has_productivity_counter, "Should find counter-evidence for productivity claim"
    assert has_cost_counter, "Should find counter-evidence for cost claim"

def test_counter_researcher_validates_sources(counter_researcher, simple_claim):
    """Test that CounterResearcher validates and filters sources."""
    counter_evidence = counter_researcher.find_counter_evidence(simple_claim)
    
    for evidence in counter_evidence:
        # Source should be a valid URL or reference
        assert evidence["source"].startswith(("http", "https", "doi", "ISBN", "PMID"))
        
        # Source should have credibility score
        assert "credibility" in evidence
        assert 0 <= evidence["credibility"] <= 1

def test_counter_researcher_handles_empty_input(counter_researcher):
    """Test that CounterResearcher gracefully handles empty or None input."""
    empty_claim = {
        "claim": "",
        "confidence": 0.0,
        "context": "",
        "source_text": ""
    }
    
    none_claim = {
        "claim": None,
        "confidence": None,
        "context": None,
        "source_text": None
    }
    
    # Should return empty list for empty input
    assert counter_researcher.find_counter_evidence(empty_claim) == []
    assert counter_researcher.find_counter_evidence(none_claim) == []

def test_counter_researcher_ranks_evidence(counter_researcher, simple_claim):
    """Test that CounterResearcher ranks evidence by relevance and credibility."""
    counter_evidence = counter_researcher.find_counter_evidence(simple_claim)
    
    if len(counter_evidence) > 1:
        # Evidence should be sorted by combined relevance and credibility
        for i in range(len(counter_evidence) - 1):
            current_score = counter_evidence[i]["relevance"] * counter_evidence[i]["credibility"]
            next_score = counter_evidence[i+1]["relevance"] * counter_evidence[i+1]["credibility"]
            assert current_score >= next_score, "Evidence should be sorted by score"
