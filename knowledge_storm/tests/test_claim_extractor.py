import pytest
from typing import List
import time
import dspy
import os
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
    lm = dspy.LM("gpt-4o-mini")
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
    # Save current LM configuration and environment
    original_lm = dspy.settings.lm
    original_api_key = os.getenv("OPENAI_API_KEY")
    original_litellm_key = os.getenv("LITELLM_API_KEY")
    
    try:
        # Clear DSPy settings and environment
        dspy.settings.lm = None
        for key in ["OPENAI_API_KEY", "LITELLM_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
        
        # Clear any cached LM instance
        if hasattr(dspy, '_lm_instance'):
            delattr(dspy, '_lm_instance')
        
        # Verify error is raised
        with pytest.raises(ValueError) as exc_info:
            extractor = ClaimExtractor()
            
        assert "No language model configured" in str(exc_info.value)
    
    finally:
        # Restore original configuration
        dspy.settings.lm = original_lm
        if original_api_key:
            os.environ["OPENAI_API_KEY"] = original_api_key
        if original_litellm_key:
            os.environ["LITELLM_API_KEY"] = original_litellm_key

def test_claim_extractor_with_research_article(claim_extractor):
    """Test that ClaimExtractor can handle real research articles."""
    with open("knowledge_storm/tests/test_data/sample_research_article.md", "r") as f:
        article_text = f.read()
    
    claims = claim_extractor.extract_claims(article_text)
    
    # Verify we got claims from different sections
    assert len(claims) >= 10  # Should find multiple claims per section
    
    # Verify market size claims
    market_claims = [c for c in claims if "billion" in c.text.lower()]
    assert len(market_claims) >= 2
    assert any(
        "$12.5 billion" in c.text and "2024" in c.text
        for c in market_claims
    )
    assert any(
        "$50 billion" in c.text and "2028" in c.text
        for c in market_claims
    )
    
    # Verify percentage claims
    percentage_claims = [c for c in claims if "%" in c.text]
    assert len(percentage_claims) >= 5
    assert any(
        "32%" in c.text and "CAGR" in c.text
        for c in percentage_claims
    )
    assert any(
        "55%" in c.text and "productivity" in c.text.lower()
        for c in percentage_claims
    )
    
    # Verify claims have proper context
    for claim in claims:
        assert claim.context in [
            "Market Growth and Adoption",
            "Technical Capabilities",
            "Developer Impact",
            "Challenges and Limitations",
            "Future Trends"
        ]
        assert claim.confidence > 0.7  # All claims should be fairly confident
        assert len(claim.source_text) > 0  # Should have source text

def test_claim_extractor_with_demo_articles(claim_extractor):
    """Test ClaimExtractor with all articles in DEMO_WORKING_DIR."""
    demo_dir = "/home/arsi/git/storm/frontend/demo_light/DEMO_WORKING_DIR"
    
    # Stats for reporting
    total_articles = 0
    total_claims = 0
    articles_with_no_claims = []
    articles_with_errors = []
    
    # Process each subdirectory
    for topic_dir in os.listdir(demo_dir):
        full_topic_dir = os.path.join(demo_dir, topic_dir)
        if not os.path.isdir(full_topic_dir):
            continue
            
        # Look for article files
        article_files = [
            f for f in os.listdir(full_topic_dir)
            if f.endswith(('.md', '.txt')) and 'article' in f.lower()
        ]
        
        for article_file in article_files:
            total_articles += 1
            article_path = os.path.join(full_topic_dir, article_file)
            
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"\n=== Processing article: {article_file} ===")
                claims = claim_extractor.extract_claims(content)
                
                if not claims:
                    articles_with_no_claims.append(article_file)
                    print(f"Warning: No claims found in {article_file}")
                else:
                    total_claims += len(claims)
                    print(f"\nFound {len(claims)} claims:")
                    
                    # Print and verify each claim
                    for i, claim in enumerate(claims, 1):
                        print(f"\nClaim {i}:")
                        print(f"Text: {claim.text}")
                        print(f"Confidence: {claim.confidence:.2f}")
                        print(f"Context: {claim.context}")
                        print(f"Source: {claim.source_text}")
                        
                        # Verify claim properties
                        assert isinstance(claim.text, str) and len(claim.text) > 0
                        assert isinstance(claim.confidence, (int, float)) and 0 <= claim.confidence <= 1
                        assert isinstance(claim.source_text, str) and len(claim.source_text) > 0
                        assert isinstance(claim.context, str) and len(claim.context) > 0
                        
            except Exception as e:
                articles_with_errors.append((article_file, str(e)))
                print(f"Error processing {article_file}: {str(e)}")
    
    # Print summary
    print("\n=== Claim Extraction Summary ===")
    print(f"Total articles processed: {total_articles}")
    print(f"Total claims found: {total_claims}")
    print(f"Average claims per article: {total_claims/total_articles if total_articles else 0:.1f}")
    
    if articles_with_no_claims:
        print("\nArticles with no claims:")
        for article in articles_with_no_claims:
            print(f"- {article}")
    
    if articles_with_errors:
        print("\nArticles with errors:")
        for article, error in articles_with_errors:
            print(f"- {article}: {error}")
    
    # Basic assertions
    assert total_articles > 0, "No articles were processed"
    assert total_claims > 0, "No claims were found in any article"
    assert len(articles_with_errors) == 0, f"{len(articles_with_errors)} articles had processing errors"
