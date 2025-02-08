# Re-Research Feature Documentation

## Overview
Re-research is a feature that enables users to generate fresh research articles on previously covered topics by:
- Excluding previously used sources
- Exploring new perspectives and angles
- Finding more recent developments
- Discovering contrasting viewpoints

## Implementation Architecture

### 1. User Interface
- Checkbox to enable re-research mode
- File uploader for previous article (markdown format)
- Enhanced topic input field with support for specific research direction guidance

### 2. Backend Components
- URL extraction from previous article
- Source exclusion system
- Enhanced LLM prompting for new perspectives
- Result filtering and validation

### 3. Data Flow
1. User uploads previous article
2. System extracts and stores URLs to exclude
3. Previous content analyzed for key themes
4. Search queries modified to target new angles
5. Results filtered to exclude previous sources
6. LLM guided to explore fresh perspectives

## Technical Implementation

### URL Extraction and Management
```python
# Enhanced URL pattern matching
url_pattern = (
    r'(?<!!)\[.*?\]\((http[s]?://[^\)]+)\)'  # Markdown links
    r'|(http[s]?://\S+)'                      # Raw URLs
)

# Optimized URL lookup
excluded_urls_set = set(excluded_urls)
search_results = [r for r in results if r.url not in excluded_urls_set]
```

### LLM Guidance System
```python
system_prompt = f"""You are an expert researcher. Your task is to explore new perspectives about: {topic}

Previous research covered:
{previous_article_summary}

Focus on:
- New developments and findings
- Alternative viewpoints
- Different methodological approaches
- Cross-disciplinary insights
- Recent updates and changes

Avoid:
- Previously used sources
- Similar analytical approaches
- Redundant examples or cases
"""
```

### Security and Performance
```python
# File validation
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_TYPES = ["text/markdown", "text/plain"]

# Performance optimization
def process_previous_article(content):
    summary = summarize_key_points(content)
    urls = extract_urls(content)
    return {
        "summary": summary,
        "excluded_urls": set(urls)
    }
```

## Edge Cases and Error Handling

1. File Processing
   - Large files (>5MB)
   - Invalid markdown
   - Non-text files
   - Unicode/special characters

2. URL Management
   - Malformed URLs
   - Redirects
   - Dead links
   - Different URL formats for same source

3. Content Analysis
   - Non-English content
   - Technical content with formulas
   - Content with embedded media

## Performance Considerations

1. URL Lookup Optimization
   - Use sets for O(1) lookups
   - Cache frequently checked URLs
   - Batch URL validation

2. Content Processing
   - Summarize large articles
   - Process in chunks
   - Cache processed results

3. Search Optimization
   - Parallel search queries
   - Prioritize recent sources
   - Smart query reformulation

## Future Improvements

1. Enhanced Analysis
   - Topic clustering for better coverage
   - Sentiment analysis for viewpoint diversity
   - Citation graph analysis

2. User Experience
   - Progress indicators
   - Source comparison views
   - Interactive guidance

3. Quality Metrics
   - Novelty scoring
   - Source diversity measurement
   - Content overlap detection

## Benefits

1. Research Quality
   - Broader perspective coverage
   - More diverse sources
   - Updated information

2. User Experience
   - Streamlined research process
   - Clear progress tracking
   - Better result differentiation

3. System Performance
   - Efficient resource usage
   - Optimized search patterns
   - Smart content reuse

## Monitoring and Metrics

1. Usage Statistics
   - Feature adoption rate
   - Success/failure ratios
   - Processing times

2. Quality Metrics
   - Source diversity score
   - Content novelty rating
   - User satisfaction

3. Performance Tracking
   - Response times
   - Resource usage
   - Error rates

## Conclusion

The re-research feature significantly enhances STORM's research capabilities by:
1. Ensuring source diversity
2. Promoting new perspectives
3. Keeping content current
4. Supporting thorough investigation

This feature aligns with academic research best practices and supports continuous learning and knowledge expansion.
