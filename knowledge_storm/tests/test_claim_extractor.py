import pytest
from typing import List
import time
import dspy
from knowledge_storm.storm_wiki.modules.claim_extraction import ClaimExtractor

@pytest.fixture
def simple_article_text():
    return """
    # Climate Change Impact
    Recent studies indicate that global temperatures have risen by 1.1°C since pre-industrial times.
    This warming has led to significant ice loss in the Arctic region.
    
    # Renewable Energy Solutions
    Solar panel efficiency has improved by 40% in the last decade.
    However, some experts suggest that wind power might be more cost-effective.
    """

@pytest.fixture
def vague_article_text():
    return """
    # General Opinions
    Many people believe renewable energy is good for the environment.
    The future of transportation might be electric.
    
    # Mixed Content
    While electric cars are becoming popular, some say they're too expensive.
    A recent study found that EV sales increased by 55% in 2023.
    """

@pytest.fixture
def malformed_article_text():
    return """
    #Malformed Header1
    This is some text without proper spacing
    #Another Bad Header
    More text without proper formatting
    Some numbers: 42% increase
    """

@pytest.fixture
def large_article_text():
    sections = []
    for i in range(10):
        sections.append(f"""
        # Section {i+1}
        In 2023, research showed a {i*10}% increase in efficiency.
        Another study demonstrated {i*5}% cost reduction.
        However, some experts disagree with these findings.
        """)
    return "\n".join(sections)

@pytest.fixture(scope="session", autouse=True)
def dspy_settings():
    """Global DSPy configuration for tests"""
    lm = dspy.LM("openai/gpt-3.5-turbo")
    dspy.configure(lm=lm)

@pytest.fixture
def claim_extractor(dspy_settings):
    """Test fixture providing configured ClaimExtractor"""
    return ClaimExtractor(max_workers=2)

def test_claim_extractor_identifies_verifiable_claims(claim_extractor, simple_article_text):
    """Test that ClaimExtractor can identify concrete, verifiable claims."""
    claims = claim_extractor.extract_claims(simple_article_text)
    
    assert len(claims) > 0
    # First claim should be about temperature rise (most specific)
    assert any(
        "1.1°C" in claim.text and "pre-industrial" in claim.text 
        for claim in claims
    )
    # Second claim should be about solar panels
    assert any(
        "40%" in claim.text and "solar panel" in claim.text.lower() 
        for claim in claims
    )

def test_claim_extractor_filters_vague_statements(claim_extractor, vague_article_text):
    """Test that ClaimExtractor filters out vague or unverifiable statements."""
    claims = claim_extractor.extract_claims(vague_article_text)
    
    # Should only extract the EV sales claim
    assert len(claims) == 1
    claim = claims[0]
    assert "55%" in claim.text
    assert any("ev sales" in c.text.lower() for c in claims)
    assert claim.confidence > 0.8  # High confidence for specific numerical claim

def test_claim_extractor_handles_malformed_input(claim_extractor, malformed_article_text):
    """Test that ClaimExtractor can handle malformed markdown and still extract claims."""
    claims = claim_extractor.extract_claims(malformed_article_text)
    
    # Should still find the numerical claim
    assert len(claims) > 0
    assert any("42%" in claim.text for claim in claims)
    
    # Context should be preserved even with malformed headers
    claim = next(claim for claim in claims if "42%" in claim.text)
    assert "Bad Header" in claim.context

def test_claim_extractor_handles_empty_input(claim_extractor):
    """Test that ClaimExtractor gracefully handles empty or None input."""
    # Empty string
    assert claim_extractor.extract_claims("") == []
    
    # Just whitespace
    assert claim_extractor.extract_claims("   \n   \t   ") == []
    
    # Just headers
    assert claim_extractor.extract_claims("# Header 1\n## Header 2") == []

def test_claim_extractor_parallel_performance(claim_extractor, large_article_text):
    """Test that ClaimExtractor can handle multiple sections efficiently."""
    # Measure execution time
    start_time = time.time()
    claims = claim_extractor.extract_claims(large_article_text)
    end_time = time.time()
    
    # Verify we got claims from multiple sections
    assert len(claims) >= 10  # At least one claim per section
    
    # Check that claims have different contexts
    contexts = {claim.context for claim in claims}
    assert len(contexts) >= 5  # Claims from at least 5 different sections
    
    # Verify numerical claims were extracted
    percentages = [
        int(num.strip('%'))
        for claim in claims
        for num in claim.text.split()
        if num.strip('%').isdigit()
    ]
    assert len(percentages) >= 10  # At least 10 percentage claims
    
    # Performance check - should process in reasonable time
    processing_time = end_time - start_time
    assert processing_time < 30  # Should process in under 30 seconds

def test_claim_extractor_no_lm_error():
    """Test that ClaimExtractor raises appropriate error when no LM is configured."""
    # Save current LM configuration
    original_lm = dspy.settings.lm
    
    try:
        # Clear DSPy settings
        dspy.settings.lm = None
        
        # Verify error is raised
        with pytest.raises(ValueError) as exc_info:
            ClaimExtractor()
        assert "No language model configured" in str(exc_info.value)
    finally:
        # Restore original settings
        dspy.settings.lm = original_lm
