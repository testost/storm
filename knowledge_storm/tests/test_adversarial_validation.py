"""Integration tests for the Adversarial Validation System."""

import json
import os
import pytest
import dspy
from typing import Dict, List

from knowledge_storm.storm_wiki.modules.adversarial_validation import AdversarialValidator

class MockRetrieve(dspy.Module):
    """Mock retrieval module for testing."""
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        
    def forward(self, query):
        """Return mock search results based on query content."""
        if "developer productivity" in query.lower():
            return [
                dspy.Prediction({
                    "text": "While AI tools can enhance productivity, studies show the impact varies greatly by team and context. Some teams report only 10-15% productivity gains, with significant time spent on tool configuration and maintenance.",
                    "source": "https://example.com/ai-productivity-study"
                })
            ]
        elif "gpt-4" in query.lower():
            return [
                dspy.Prediction({
                    "text": "Independent evaluations of GPT-4's coding abilities show more modest results. When considering real-world programming tasks with complex requirements, GPT-4's success rate drops to 45-55%, comparable to mid-level developers.",
                    "source": "doi:10.1234/ai.2024.789"
                })
            ]
        elif "testing" in query.lower():
            return [
                dspy.Prediction({
                    "text": "Research challenges the 80% edge case coverage claim. Studies indicate AI testing tools primarily find common edge cases, with novel or complex scenarios still requiring human expertise. Actual unique edge case coverage is closer to 40-50%.",
                    "source": "https://example.com/ai-testing-analysis"
                })
            ]
        elif "cost" in query.lower() or "roi" in query.lower():
            return [
                dspy.Prediction({
                    "text": "Cost savings from AI tools are often overestimated. A comprehensive industry survey reveals that most organizations achieve 20-30% cost reduction in the first year, with high implementation and training costs offsetting gains.",
                    "source": "doi:10.1234/software.2024.456"
                })
            ]
        elif "suomalaiset" in query.lower() or "finnish" in query.lower():
            return [
                dspy.Prediction({
                    "text": "While Finnish metal bands have seen success, the 80% international awards claim is disputed. Recent analysis shows Finnish bands account for approximately 40-45% of Nordic metal awards.",
                    "source": "https://example.com/nordic-metal-analysis"
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
def demo_dir(tmp_path):
    """Create a temporary demo directory structure."""
    # Create English article
    en_article_dir = tmp_path / "Current_state_of_AI_assisted_coding"
    en_article_dir.mkdir()
    with open(en_article_dir / "storm_gen_article.txt", "w", encoding="utf-8") as f:
        f.write("""# The Impact of AI on Software Development

## Introduction
Artificial Intelligence (AI) has revolutionized software development in recent years. 
Studies indicate that AI-powered tools can increase developer productivity by 40% 
while reducing bugs by 30%.

## Code Generation
Large Language Models (LLMs) like GPT-4 have shown remarkable capabilities in code 
generation. Recent benchmarks demonstrate that GPT-4 achieves a 95% success rate 
in solving programming challenges, significantly outperforming human developers who 
average 65% success rate.

## Testing and Quality Assurance
AI-driven testing tools have transformed QA processes. Automated test generation 
using AI can cover 80% of edge cases that human testers might miss. Companies 
adopting AI testing report a 50% reduction in post-release defects.

## Cost Implications
While AI tools require significant upfront investment, the long-term ROI is 
substantial. Organizations report an average 60% reduction in development costs 
within the first year of adopting AI-powered development tools.""")
            
    # Create Finnish article
    fi_article_dir = tmp_path / "Suomen_parhaat_bändit"
    fi_article_dir.mkdir()
    with open(fi_article_dir / "storm_gen_article.txt", "w", encoding="utf-8") as f:
        f.write("""# Suomen parhaat bändit

## Johdanto
Suomalainen musiikkiteollisuus on tuottanut monia kansainvälisesti menestyneitä 
yhtyeitä. Tutkimusten mukaan suomalaiset bändit ovat saavuttaneet 45% kasvun 
kansainvälisissä kuuntelukerroissa viimeisen viiden vuoden aikana.

## Metallimusiikin vaikutus
Suomi tunnetaan erityisesti metallimusiikin suurvaltana. Tilastojen mukaan 
suomalaiset metallibändit tuottavat 30% kaikesta pohjoismaisesta metallimusiikista 
ja keräävät yli 80% alan kansainvälisistä palkinnoista.

## Digitaalinen menestys
Suomalaisten artistien digitaaliset striimausmäärät ovat kasvaneet merkittävästi. 
Vuonna 2024 suomalaiset yhtyeet keräsivät keskimäärin 50 miljoonaa striimausta 
kuukaudessa, mikä on 75% enemmän kuin vuonna 2020.""")
            
    return tmp_path

@pytest.fixture
def validator(dspy_settings, demo_dir):
    """Test fixture providing configured AdversarialValidator."""
    validator = AdversarialValidator(demo_dir)
    validator.counter_researcher.retrieve = MockRetrieve()
    return validator

def test_validate_english_article(validator):
    """Test adversarial validation on English article."""
    article_name = "Current_state_of_AI_assisted_coding"
    
    # Run validation
    validated_claims = validator.validate_article(article_name)
    
    # Verify claims were saved
    claims_path = validator.get_claims_path(article_name)
    assert os.path.exists(claims_path)
    with open(claims_path, "r", encoding="utf-8") as f:
        saved_claims = json.load(f)
    assert len(saved_claims) >= 4
    
    # Verify counter-evidence was saved
    counter_evidence_path = validator.get_counter_evidence_path(article_name)
    assert os.path.exists(counter_evidence_path)
    with open(counter_evidence_path, "r", encoding="utf-8") as f:
        saved_results = json.load(f)
    assert len(saved_results) >= 4
    
    # Verify specific counter-arguments
    productivity_claim = next(
        (vc for vc in saved_claims 
         if "productivity" in vc["claim"].lower()),
        None
    )
    assert productivity_claim, "Should find productivity-related claim"

def test_validate_finnish_article(validator):
    """Test adversarial validation on Finnish article."""
    article_name = "Suomen_parhaat_bändit"
    
    # Run validation
    validated_claims = validator.validate_article(article_name)
    
    # Verify claims were saved
    claims_path = validator.get_claims_path(article_name)
    assert os.path.exists(claims_path)
    with open(claims_path, "r", encoding="utf-8") as f:
        saved_claims = json.load(f)
    assert len(saved_claims) >= 3
    
    # Verify counter-evidence was saved
    counter_evidence_path = validator.get_counter_evidence_path(article_name)
    assert os.path.exists(counter_evidence_path)
    with open(counter_evidence_path, "r", encoding="utf-8") as f:
        saved_results = json.load(f)
    assert len(saved_results) >= 3
    
    # Verify we found Finnish claims
    assert any(
        "suomalaiset" in c["claim"].lower()
        for c in saved_claims
    ), "Should find claims about Finnish music"
