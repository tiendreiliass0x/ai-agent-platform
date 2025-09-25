# RAG System Evaluation Framework

A comprehensive evaluation system for testing and monitoring your RAG (Retrieval-Augmented Generation) pipeline.

## Overview

This evaluation framework provides:

- **Retrieval Evaluation**: Measures how well your system finds relevant documents
- **Response Quality**: Evaluates the final AI responses using multiple metrics
- **End-to-End Testing**: Complete pipeline evaluation including performance metrics
- **Continuous Monitoring**: Production monitoring with alerting
- **Hallucination Detection**: Identifies when AI generates unsupported information

## Quick Start

### 1. Basic Evaluation

```bash
# Run comprehensive evaluation
cd backend
python scripts/run_evaluation.py --type comprehensive --agent-id 1

# Run only retrieval evaluation
python scripts/run_evaluation.py --type retrieval --agent-id 1

# Run only response quality evaluation
python scripts/run_evaluation.py --type response --agent-id 1
```

### 2. Load Testing

```bash
# Test with 10 concurrent requests
python scripts/run_evaluation.py --type comprehensive --concurrent 10
```

### 3. Continuous Monitoring

```bash
# Monitor every 5 minutes (300 seconds)
python scripts/run_evaluation.py --type monitor --monitor-interval 300
```

## Evaluation Metrics

### Retrieval Metrics

- **Precision@K**: Fraction of retrieved chunks that are relevant
- **Recall@K**: Fraction of relevant chunks that were retrieved
- **F1@K**: Harmonic mean of precision and recall
- **NDCG@K**: Normalized Discounted Cumulative Gain (considers ranking)
- **MRR**: Mean Reciprocal Rank of first relevant result
- **Hit Rate**: Whether any relevant chunk was found

### Response Quality Metrics

- **Relevance**: How well the response addresses the query
- **Accuracy**: Factual correctness of the information
- **Completeness**: Whether all parts of the question are answered
- **Clarity**: How clear and understandable the response is
- **Helpfulness**: How useful the response is to the user

### System Performance Metrics

- **Response Time**: Average, 95th, and 99th percentile response times
- **Success Rate**: Percentage of successful requests
- **Throughput**: Requests processed per second
- **Error Rate**: Percentage of failed requests

## Setting Up Test Cases

### Manual Test Cases

Create test cases in your evaluation scripts:

```python
from app.evaluation.end_to_end_evaluator import E2ETestCase

test_cases = [
    E2ETestCase(
        query="What is your refund policy?",
        expected_chunks=["refund-policy-section"],
        expected_response="Our refund policy allows...",
        context="Customer service inquiry",
        difficulty="easy"
    ),
    # Add more test cases...
]
```

### Auto-Generated Test Cases

Use the built-in test case generation:

```python
from app.evaluation.response_evaluator import ResponseEvaluator

evaluator = ResponseEvaluator(rag_service)
test_cases = await evaluator.generate_test_cases(
    agent_context="Customer support for e-commerce platform",
    num_cases=20
)
```

## Understanding Results

### Sample Output

```
🔍 COMPREHENSIVE RAG EVALUATION REPORT
============================================================
Agent ID: 1
Evaluation Time: 45.23s
Test Cases: 15

⚡ SYSTEM PERFORMANCE:
  Avg Response Time: 2.34s
  95th Percentile: 4.12s
  Success Rate: 98.0%
  Throughput: 12.5 req/s

🎯 RETRIEVAL PERFORMANCE:
  Precision@5: 0.756
  Recall@5: 0.823
  F1@5: 0.788

🤖 RESPONSE QUALITY:
  Accuracy: 0.834
  Helpfulness: 0.791
  Overall: 0.812

👥 USER EXPERIENCE SCORE: 0.798

💡 RECOMMENDATIONS:
  🎯 Low retrieval accuracy (<70%). Consider improving document chunking.
  📝 Low response accuracy (<70%). Review system prompts.
```

### Interpreting Scores

- **0.8+ (Excellent)**: Production-ready performance
- **0.7-0.8 (Good)**: Acceptable with minor improvements needed
- **0.6-0.7 (Fair)**: Needs optimization before production
- **<0.6 (Poor)**: Significant improvements required

## Production Monitoring

### Setting Up Alerts

The monitoring system can detect performance degradation:

```python
# In your monitoring script
if overall_score < 0.7:
    send_alert(f"RAG performance degraded: {overall_score:.3f}")
```

### Recommended Monitoring Schedule

- **Development**: Run comprehensive evaluation before each deployment
- **Staging**: Continuous monitoring every 30 minutes
- **Production**: Continuous monitoring every 5-10 minutes

## Improving Performance

### Common Issues and Solutions

1. **Low Retrieval Precision**
   - Improve document chunking strategy
   - Fine-tune embedding models
   - Add better metadata to chunks

2. **Slow Response Times**
   - Optimize vector database queries
   - Use faster embedding models
   - Implement response caching

3. **Low Response Quality**
   - Improve system prompts
   - Use better language models
   - Add more context to prompts

4. **High Hallucination Rate**
   - Stricter prompt instructions
   - Better source attribution
   - Lower model temperature

## Integration with CI/CD

Add evaluation to your deployment pipeline:

```yaml
# .github/workflows/evaluation.yml
- name: Run RAG Evaluation
  run: |
    cd backend
    python scripts/run_evaluation.py --type comprehensive --agent-id 1
    # Parse results and fail if scores are too low
```

## Advanced Usage

### Custom Metrics

Extend the evaluation framework with your own metrics:

```python
class CustomEvaluator(ResponseEvaluator):
    def custom_metric(self, query, response):
        # Your custom evaluation logic
        return score
```

### Benchmark Datasets

Create domain-specific benchmark datasets for your use case:

```python
# Create golden dataset for your domain
golden_dataset = [
    {
        "query": "domain-specific question",
        "expected_chunks": ["relevant-chunk-ids"],
        "expected_response": "ideal response"
    }
]
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the backend directory
2. **API Key Issues**: Check your OpenAI API key configuration
3. **Database Connection**: Ensure your database is running for agent data
4. **Memory Issues**: Reduce batch size for large evaluations

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To add new evaluation metrics:

1. Create new metric functions
2. Add to appropriate evaluator class
3. Update aggregation logic
4. Add tests and documentation

## License

Part of the AI Agent Platform - see main project license.