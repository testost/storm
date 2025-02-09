"""Adversarial Validation System for STORM."""

import json
import os
from typing import Dict, List, Optional

from .claim_extraction import ClaimExtractor
from .counter_research import CounterResearcher

class AdversarialValidator:
    """Adversarial validation system that combines claim extraction and counter-evidence research."""
    
    def __init__(self, working_dir: str):
        """Initialize the adversarial validator.
        
        Args:
            working_dir: Base directory containing article subdirectories
        """
        self.working_dir = working_dir
        self.claim_extractor = ClaimExtractor()
        self.counter_researcher = CounterResearcher()
        
    def get_article_path(self, article_name: str) -> str:
        """Get the full path to an article directory.
        
        Args:
            article_name: Name of the article directory
            
        Returns:
            Absolute path to the article directory
        """
        return os.path.join(self.working_dir, article_name)
        
    def get_claims_path(self, article_name: str) -> str:
        """Get the path to the claims file for an article.
        
        Args:
            article_name: Name of the article directory
            
        Returns:
            Path to the claims.json file
        """
        claims_dir = os.path.join(self.get_article_path(article_name), "adversarial_validation")
        os.makedirs(claims_dir, exist_ok=True)
        return os.path.join(claims_dir, "claims.json")
        
    def get_counter_evidence_path(self, article_name: str) -> str:
        """Get the path to the counter-evidence file for an article.
        
        Args:
            article_name: Name of the article directory
            
        Returns:
            Path to the counter_evidence.json file
        """
        counter_evidence_dir = os.path.join(self.get_article_path(article_name), "adversarial_validation")
        os.makedirs(counter_evidence_dir, exist_ok=True)
        return os.path.join(counter_evidence_dir, "counter_evidence.json")
        
    def extract_and_save_claims(self, article_name: str) -> List[Dict]:
        """Extract claims from an article and save them to disk.
        
        Args:
            article_name: Name of the article directory
            
        Returns:
            List of extracted claims
        """
        # Read the article text
        article_path = os.path.join(self.get_article_path(article_name), "storm_gen_article.txt")
        with open(article_path, "r", encoding="utf-8") as f:
            article_text = f.read()
            
        # Extract claims
        claims = self.claim_extractor.extract_claims(article_text)
        
        # Convert claims to JSON-serializable format
        serializable_claims = []
        for claim in claims:
            serializable_claims.append({
                "claim": claim.text,
                "confidence": claim.confidence,
                "source_text": claim.source_text,
                "context": claim.context
            })
            
        # Save claims
        claims_path = self.get_claims_path(article_name)
        with open(claims_path, "w", encoding="utf-8") as f:
            json.dump(serializable_claims, f, ensure_ascii=False, indent=2)
            
        return serializable_claims
        
    def research_and_save_counter_evidence(self, article_name: str, claims: Optional[List[Dict]] = None) -> List[Dict]:
        """Find counter-evidence for claims and save results to disk.
        
        Args:
            article_name: Name of the article directory
            claims: Optional list of claims. If not provided, will load from disk.
            
        Returns:
            List of claims with their counter-evidence
        """
        # Load claims if not provided
        if claims is None:
            claims_path = self.get_claims_path(article_name)
            with open(claims_path, "r", encoding="utf-8") as f:
                claims = json.load(f)
                
        # Find counter-evidence for each claim
        validated_claims = []
        for claim in claims:
            counter_evidence = self.counter_researcher.find_counter_evidence(claim)
            validated_claims.append({
                "claim": claim,
                "counter_evidence": counter_evidence
            })
            
        # Save results
        counter_evidence_path = self.get_counter_evidence_path(article_name)
        with open(counter_evidence_path, "w", encoding="utf-8") as f:
            json.dump(validated_claims, f, ensure_ascii=False, indent=2)
            
        return validated_claims
        
    def validate_article(self, article_name: str) -> List[Dict]:
        """Run the complete validation workflow for an article.
        
        Args:
            article_name: Name of the article directory
            
        Returns:
            List of claims with their counter-evidence
        """
        claims = self.extract_and_save_claims(article_name)
        return self.research_and_save_counter_evidence(article_name, claims)
