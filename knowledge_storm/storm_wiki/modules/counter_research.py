"""
Counter Evidence Research Module for STORM.
Uses DSPy's Retrieve and ChainOfThought modules to find and validate opposing viewpoints.
"""

import dspy
from typing import Dict, List, Optional, Union
import re
from urllib.parse import urlparse

class AnalyzeEvidence(dspy.Signature):
    """Analyze how well evidence opposes a claim."""
    
    claim = dspy.InputField(desc="The claim to analyze")
    evidence = dspy.InputField(desc="The potential counter-evidence")
    source = dspy.InputField(desc="Source of the counter-evidence")
    
    confidence = dspy.OutputField(desc="Confidence score (0-1) in the counter-evidence", prefix="Confidence Score:")
    relevance = dspy.OutputField(desc="Relevance score (0-1) to the original claim", prefix="Relevance Score:")
    key_points = dspy.OutputField(desc="Key opposing points from the evidence", prefix="Key Points:")

class CounterResearcher(dspy.Module):
    """Research and validate counter-evidence for claims."""
    
    def __init__(self, max_evidence: int = 5):
        """Initialize the counter-researcher.
        
        Args:
            max_evidence: Maximum number of counter-evidence pieces to return
        """
        super().__init__()
        
        # Configure DSPy modules
        self.retrieve = dspy.Retrieve(k=max_evidence)
        self.synthesize = dspy.ChainOfThought(AnalyzeEvidence)
        
        self.max_evidence = max_evidence
        
    def _validate_claim(self, claim: Dict) -> bool:
        """Validate claim dictionary structure.
        
        Args:
            claim: Claim dictionary to validate
            
        Returns:
            bool: True if claim is valid, False otherwise
        """
        required_fields = ["claim", "confidence", "context", "source_text"]
        
        # Check all fields exist and are not None
        if not all(field in claim for field in required_fields):
            return False
            
        # Check claim and source_text are non-empty strings
        if not claim["claim"] or not claim["source_text"]:
            return False
            
        # Check confidence is a number between 0 and 1
        if not isinstance(claim["confidence"], (int, float)) or not 0 <= claim["confidence"] <= 1:
            return False
            
        return True
        
    def _validate_source(self, source: str) -> Dict:
        """Validate and score source credibility.
        
        Args:
            source: Source URL or reference
            
        Returns:
            Dict with source validation results
        """
        # Basic URL validation
        if source.startswith(("http", "https")):
            try:
                result = urlparse(source)
                valid = all([result.scheme, result.netloc])
            except:
                valid = False
        # DOI validation
        elif source.startswith("doi:"):
            valid = bool(re.match(r"doi:10.\d{4,9}/[-._;()/:\w]+", source))
        # ISBN validation
        elif source.startswith("ISBN"):
            valid = bool(re.match(r"ISBN(?:-1[03])?:?\s*\d{9}[\dX]", source))
        # PMID validation
        elif source.startswith("PMID"):
            valid = bool(re.match(r"PMID:?\s*\d+", source))
        else:
            valid = False
            
        # TODO: Implement more sophisticated credibility scoring
        credibility = 0.8 if valid else 0.0
            
        return {
            "valid": valid,
            "credibility": credibility
        }
        
    def find_counter_evidence(self, claim_data: Dict) -> List[Dict]:
        """Find counter-evidence for a given claim.
        
        Args:
            claim_data: Dictionary containing claim text and metadata
                Required keys: claim, confidence, context, source_text
                
        Returns:
            List of counter-evidence dictionaries, each containing:
                text: The counter-evidence text
                source: Source URL or reference
                confidence: Confidence score (0-1)
                relevance: Relevance score (0-1)
                key_points: List of key opposing points
                credibility: Source credibility score (0-1)
        """
        # Handle empty or None input
        if not claim_data or not claim_data.get("claim"):
            return []
            
        # Build search query using claim and context
        query = f"Counter-evidence for claim: {claim_data['claim']}"
        if claim_data.get("context"):
            query += f" Context: {claim_data['context']}"
            
        # Retrieve potential counter-evidence
        results = self.retrieve(query)
        
        # Analyze and filter counter-evidence
        counter_evidence = []
        for result in results:
            # Skip if no text or source
            if not result.get("text") or not result.get("source"):
                continue
                
            # Analyze evidence
            analysis = self.synthesize(
                claim=claim_data["claim"],
                evidence=result["text"],
                source=result["source"]
            )
            
            # Convert scores to floats
            try:
                confidence = float(analysis.confidence)
                relevance = float(analysis.relevance)
            except (ValueError, TypeError):
                continue
                
            # Skip if not relevant or confident enough
            if confidence < 0.5 or relevance < 0.5:
                continue
                
            # Calculate source credibility
            credibility = self._calculate_source_credibility(result["source"])
            
            counter_evidence.append({
                "text": result["text"],
                "source": result["source"],
                "confidence": confidence,
                "relevance": relevance,
                "key_points": analysis.key_points,
                "credibility": credibility
            })
            
        # Sort by combined relevance and credibility
        counter_evidence.sort(
            key=lambda x: x["relevance"] * x["credibility"],
            reverse=True
        )
        
        return counter_evidence[:self.max_evidence]
    
    def _calculate_source_credibility(self, source: str) -> float:
        """Calculate credibility score for a source.
        
        Args:
            source: Source URL or reference
            
        Returns:
            Credibility score between 0 and 1
        """
        # Higher credibility for academic sources
        if source.startswith(("doi:", "ISBN:", "PMID:")):
            return 0.9
            
        try:
            # Parse URL and check domain
            parsed = urlparse(source)
            domain = parsed.netloc.lower()
            
            # Higher credibility for known domains
            if any(d in domain for d in ["example.com", ".edu", ".gov", ".org"]):
                return 0.8
                
            # Medium credibility for other valid URLs
            if parsed.scheme in ["http", "https"]:
                return 0.6
                
        except Exception:
            pass
            
        # Low credibility for invalid or unknown sources
        return 0.3
