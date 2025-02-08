# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added debug logging in ClaimExtractor to help diagnose section parsing and claim extraction issues

### Changed
- Updated DSPy configuration to use new `dspy.LM` class instead of deprecated `dspy.OpenAI`
- Enhanced ExtractClaims signature docstring to better guide model in extracting numerical claims
- Improved malformed section handling in ClaimExtractor's `_split_into_sections` method
- Updated test assertions to use case-insensitive string comparisons

### Fixed
- Fixed handling of malformed markdown headers (no space after #)
- Fixed claim extraction from sections with numerical content
- Fixed DSPy configuration to work with latest version (2.5+)

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
