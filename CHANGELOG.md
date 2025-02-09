# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added CounterResearcher module for finding and validating counter-evidence to claims
- Added source credibility scoring based on URL type and domain
- Added comprehensive test suite for CounterResearcher functionality
- Added debug logging in ClaimExtractor to help diagnose section parsing and claim extraction issues
- Enhanced claim extraction with more aggressive pattern matching
- Improved confidence scoring for claims (0.9 for precise numerical claims, 0.8 for specific claims, 0.7 for measurable claims)
- Expanded claim categories to include market trends, regulatory requirements, rankings, and historical events
- Better handling of multiple claims within the same sentence
- Added support for GPT-4o-mini model with improved claim extraction performance
- Added validation check for language model configuration in ClaimExtractor

### Changed
- Updated DSPy configuration to use new `dspy.LM` class instead of deprecated `dspy.OpenAI`
- Enhanced ExtractClaims signature docstring to better guide model in extracting numerical claims
- Improved malformed section handling in ClaimExtractor's `_split_into_sections` method
- Updated test assertions to use case-insensitive string comparisons
- Moved LM validation before model creation in ClaimExtractor initialization
- Improved claim extraction performance (27.4 claims/article average)

### Fixed
- Fixed handling of malformed markdown headers (no space after #)
- Fixed claim extraction from sections with numerical content
- Fixed DSPy configuration to work with latest version (2.5+)
- Fixed error handling when no language model is configured

### Deprecated
- Removed usage of deprecated DSPy OpenAI client in favor of new LM interface

### Security
- No changes

## [0.1.0] - 2025-02-09

### Added
- Initial release of Storm knowledge management system
- ClaimExtractor module for identifying verifiable claims in text
- Test suite for ClaimExtractor functionality
- Support for parallel processing of document sections
