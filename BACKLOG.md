# Storm Project Backlog

## Claim Extractor Improvements

### High Priority
1. Add support for multilingual claim extraction
   - ✅ Improved Finnish language claim detection
   - Test with other languages beyond English and Finnish
   - Consider using language-specific models or fine-tuning

2. Evaluate newer LLM models
   - ✅ Tested GPT-4o-mini for improved claim detection accuracy
   - ✅ Benchmarked against GPT-3.5 results (27.4 vs ~20 claims/article)
   - Consider open-source alternatives like Llama 2

3. Add claim validation rules
   - Verify numerical claims are within reasonable ranges
   - Check for contradictory claims in the same document
   - Validate dates and timelines for consistency

### Medium Priority
1. Expand claim categories
   - ✅ Added support for comparative claims
   - ✅ Added detection of milestone claims
   - ✅ Added identification of trend claims
   - ✅ Added extraction of temporal claims

2. Improve context extraction
   - ✅ Better section boundary detection
   - ✅ Maintain hierarchical relationships between claims
   - Link related claims across sections

3. Performance optimization
   - ✅ Optimized token usage in LLM prompts
   - Cache common claim patterns
   - Batch process similar claims

### Low Priority
1. Adjust confidence thresholds
   - ✅ Fine-tuned based on testing (0.9 for precise claims, 0.8 for specific claims)
   - ✅ Added confidence modifiers for specific claim types
   - Consider source credibility in scoring

2. Enhanced reporting
   - ✅ Generate claim extraction statistics
   - Visualize claim distribution across sections
   - ✅ Track claim extraction success rates

3. User interface improvements
   - Add claim highlighting in source text
   - Enable manual claim verification
   - Provide claim editing capabilities

## Model and Language Optimization

### High Priority
1. Language Comparison Testing
   - Create test suite with paired English-Finnish articles
   - Measure claim extraction accuracy for both languages
   - Identify patterns that work/fail in Finnish
   - Document Finnish-specific challenges (morphology, compounds)

2. LLM Model Benchmarking
   - ✅ Set up controlled test environment
   - ✅ Compare GPT-3.5 vs GPT-4o-mini performance
   - ✅ Measure metrics:
     * ✅ Claims found per article (27.4 vs ~20)
     * ✅ Confidence scores (improved precision)
     * ✅ Processing time
     * Cost per article
     * ✅ Language-specific accuracy

3. Finnish Language Optimization
   - ✅ Add Finnish-specific claim patterns
   - ✅ Optimize section parsing for Finnish text structure
   - Create Finnish test cases for common claim types
   - Consider Finnish language models (if available)

### Medium Priority
1. Model Fine-tuning Investigation
   - Evaluate cost/benefit of fine-tuning
   - Collect training data for claim extraction
   - Test fine-tuning on smaller models first
   - Compare with zero-shot performance

2. Multilingual Support Framework
   - Design language-agnostic claim patterns
   - Add language detection and routing
   - Support easy addition of new languages
   - Create language-specific confidence adjustments

3. Performance Optimization by Model
   - Implement model-specific prompts
   - Optimize token usage per model
   - Cache common patterns by language
   - Add fallback options for rate limits

### Low Priority
1. Cost Optimization
   - Track token usage by language
   - Implement model switching based on needs
   - Optimize prompt length for each model
   - Consider batching similar languages

2. Alternative Models Research
   - Evaluate open-source alternatives
   - Test specialized NLP models
   - Consider hybrid approaches
   - Monitor new model releases

3. Monitoring and Analytics
   - Track performance by language
   - Compare model effectiveness
   - Generate cost reports
   - Monitor error rates by language
