import pytest
import dspy
from dspy.teleprompt import BootstrapFewShot

@pytest.fixture(scope="session")
def dspy_settings():
    """Global DSPy configuration for tests"""
    dspy.settings.configure(
        lm=dspy.OpenAI(model="gpt-3.5-turbo"),
        tracer=BootstrapFewShot()
    )

@pytest.fixture
def claim_extractor(dspy_settings):
    """Test fixture providing configured ClaimExtractor"""
    from knowledge_storm.storm_wiki.modules.claim_extraction import ClaimExtractor
    return ClaimExtractor(max_workers=2)
