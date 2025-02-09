from dataclasses import dataclass
from typing import List, Optional
import dspy
from dspy.signatures import Signature
from dspy.signatures.field import InputField, OutputField
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import os

@dataclass
class Claim:
    text: str
    confidence: float
    source_text: str
    context: str

class ExtractClaims(Signature):
    """Signature for claim extraction.
    
    Extract verifiable claims from the given text. A verifiable claim:
    1. Makes a specific, measurable statement
    2. Often includes numbers, dates, statistics, or concrete facts
    3. Can be fact-checked using reliable sources
    4. Has clear context and scope
    
    For each claim, provide:
    - claim_text: The specific claim, focusing on numerical or statistical statements
    - confidence: Float between 0-1 indicating confidence in claim verifiability
    - source_text: Original sentence containing the claim
    
    IMPORTANT GUIDELINES:
    1. Aggressively extract ANY statement that contains:
       - Numbers (e.g., "70% of users", "$50 million in revenue")
       - Dates (e.g., "by 2025", "since 2020")
       - Statistics (e.g., "doubled in size", "grew by half")
       - Rankings (e.g., "first company to...", "largest in the industry")
       - Specific measurements (e.g., "reduced by 3 hours", "increased 2x")
    
    2. Look for claims about:
       - Market sizes and growth
       - User/customer numbers
       - Performance metrics
       - Industry rankings
       - Timeline commitments
       - Regulatory requirements
       - Technical specifications
       - Historical events
    
    3. Confidence scoring:
       - 0.9-1.0: Precise numerical claims with clear context
       - 0.7-0.9: Specific claims with some qualifiers
       - 0.5-0.7: Claims that are measurable but less precise
       - <0.5: Vague or difficult to verify claims (exclude these)
    
    4. Extract multiple claims from the same sentence if present
    """
    text: str = InputField(desc="Text to extract claims from")
    claims: List[dict] = OutputField(desc="List of dicts with claim_text, confidence, source_text")

class ClaimExtractor:
    def __init__(self, max_workers: int = 4, lm: Optional[dspy.LM] = None):
        """Initialize the claim extractor.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            lm: Optional language model instance. If not provided, uses global DSPy settings.
        """
        self.max_workers = max_workers
        
        # Validate LM configuration before doing anything else
        self._validate_lm_config(lm)
        
        # Configure default model if none provided
        if not lm:
            lm = dspy.LM("gpt-4o-mini")
            
        # Configure DSPy predictor
        self.extract = dspy.Predict(ExtractClaims)
    
    def _validate_lm_config(self, lm: Optional[dspy.LM] = None):
        """Ensure valid LM configuration exists.
        
        Args:
            lm: Optional language model instance to configure
            
        Raises:
            ValueError: If no language model is configured
        """
        try:
            if lm:
                dspy.settings.configure(lm=lm)
            elif not dspy.settings.lm and not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "No language model configured. Either:\n"
                    "1. Call dspy.configure(lm=...) globally first\n"
                    "2. Pass lm parameter when creating ClaimExtractor\n"
                    "3. Set OPENAI_API_KEY environment variable"
                )
        except Exception as e:
            raise ValueError(f"Failed to configure language model: {str(e)}")
    
    def _normalize_header(self, header: str) -> str:
        """Normalize markdown header by removing extra spaces and #."""
        return re.sub(r'^#+\s*', '', header).strip()
    
    def _split_into_sections(self, text: str) -> List[dict]:
        """Split text into sections, handling malformed markdown."""
        if not text or not text.strip():
            return []
            
        sections = []
        current_section = {"heading": "", "text": []}
        
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                # Only create new section if there's actual text in current section
                if current_section["text"]:
                    sections.append(current_section)
                # Handle malformed headers (no space after #)
                header = line.lstrip("#").strip() if line.lstrip("#").strip() else "Default Section"
                current_section = {
                    "heading": header,
                    "text": []
                }
            elif line:
                # If we haven't encountered any header yet, create a default section
                if not current_section["heading"]:
                    current_section["heading"] = "Default Section"
                current_section["text"].append(line)
        
        # Add the last section if it has text
        if current_section["text"]:
            sections.append(current_section)
            
        # If no sections were found, create a default one
        if not sections:
            sections.append({
                "heading": "Default Section",
                "text": [line for line in text.split("\n") if line.strip()]
            })
            
        # Debug logging
        print("\nSections found:")
        for section in sections:
            print(f"- Header: {section['heading']}")
            print(f"  Text: {' '.join(section['text'])}")
            
        return sections
    
    def _extract_claims_from_section(self, section: dict) -> List[Claim]:
        """Extract claims from a single section."""
        try:
            text = "\n".join(section["text"])
            print(f"\nAttempting to extract claims from section '{section['heading']}'")
            print(f"Text: {text}")
            
            result = self.extract(text=text)
            print(f"Extraction result: {result}")
            
            claims = []
            # Claims are now returned directly as a list of dictionaries
            for claim_dict in result.claims:
                claims.append(Claim(
                    text=claim_dict["claim_text"],
                    confidence=claim_dict["confidence"],
                    source_text=claim_dict["source_text"],
                    context=section["heading"]
                ))
            return claims
            
        except Exception as e:
            print(f"Error extracting claims from section '{section['heading']}': {str(e)}")
            return []
    
    def extract_claims(self, text: str) -> List[Claim]:
        """Extract verifiable claims from text.
        
        Args:
            text: Input text to extract claims from
            
        Returns:
            List of Claim objects containing the extracted claims
        """
        sections = self._split_into_sections(text)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_section = {
                executor.submit(self._extract_claims_from_section, section): section
                for section in sections
            }
            
            # Collect results
            all_claims = []
            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    claims = future.result()
                    all_claims.extend(claims)
                except Exception as e:
                    print(f"Error processing section '{section['heading']}': {str(e)}")
                    
        return all_claims
