"""
Response quality evaluation for RAG system.
Evaluates the quality of final AI responses using multiple metrics.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from datetime import datetime
import re

from ..services.rag_service import RAGService
from ..core.config import settings


@dataclass
class ResponseTestCase:
    """Test case for response evaluation"""
    query: str
    expected_response: Optional[str] = None
    context: str = ""
    evaluation_criteria: List[str] = None
    difficulty: str = "medium"


@dataclass
class ResponseMetrics:
    """Metrics for response evaluation"""
    relevance_score: float  # How relevant is the response to the query
    accuracy_score: float  # How factually accurate is the response
    completeness_score: float  # How complete is the response
    clarity_score: float  # How clear and understandable is the response
    helpfulness_score: float  # How helpful is the response to the user
    overall_score: float  # Weighted average of all scores


class ResponseEvaluator:
    """Evaluates response quality in RAG system"""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    async def evaluate_responses(
        self,
        test_cases: List[ResponseTestCase],
        agent_id: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate response quality on test cases

        Args:
            test_cases: List of test cases with queries
            agent_id: Agent to test responses for

        Returns:
            Dictionary with evaluation metrics and detailed results
        """
        results = []

        for test_case in test_cases:
            # Generate response using RAG system
            generated_response = await self.rag_service.generate_response(
                query=test_case.query,
                agent_id=agent_id,
                conversation_id=None
            )

            # Evaluate the response quality
            metrics = await self._evaluate_single_response(
                query=test_case.query,
                response=generated_response,
                expected_response=test_case.expected_response,
                context=test_case.context,
                criteria=test_case.evaluation_criteria or []
            )

            result = {
                "query": test_case.query,
                "generated_response": generated_response,
                "expected_response": test_case.expected_response,
                "difficulty": test_case.difficulty,
                "context": test_case.context,
                "metrics": metrics
            }
            results.append(result)

        # Aggregate metrics across all test cases
        aggregate_metrics = self._aggregate_response_metrics(results)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(test_cases),
            "agent_id": agent_id,
            "aggregate_metrics": aggregate_metrics,
            "per_difficulty_metrics": self._metrics_by_difficulty(results),
            "detailed_results": results
        }

    async def _evaluate_single_response(
        self,
        query: str,
        response: str,
        expected_response: Optional[str] = None,
        context: str = "",
        criteria: List[str] = None
    ) -> ResponseMetrics:
        """Evaluate a single response using LLM-based evaluation"""

        evaluation_prompt = self._build_evaluation_prompt(
            query, response, expected_response, context, criteria
        )

        try:
            # Use GPT-4 to evaluate the response
            eval_response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator of AI assistant responses. Provide objective, detailed evaluations."
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ],
                temperature=0.1
            )

            evaluation_text = eval_response.choices[0].message.content
            scores = self._parse_evaluation_scores(evaluation_text)

            return ResponseMetrics(
                relevance_score=scores.get("relevance", 0.0),
                accuracy_score=scores.get("accuracy", 0.0),
                completeness_score=scores.get("completeness", 0.0),
                clarity_score=scores.get("clarity", 0.0),
                helpfulness_score=scores.get("helpfulness", 0.0),
                overall_score=scores.get("overall", 0.0)
            )

        except Exception as e:
            print(f"Error in response evaluation: {e}")
            # Return default scores if evaluation fails
            return ResponseMetrics(
                relevance_score=0.0,
                accuracy_score=0.0,
                completeness_score=0.0,
                clarity_score=0.0,
                helpfulness_score=0.0,
                overall_score=0.0
            )

    def _build_evaluation_prompt(
        self,
        query: str,
        response: str,
        expected_response: Optional[str],
        context: str,
        criteria: List[str]
    ) -> str:
        """Build prompt for LLM-based response evaluation"""

        prompt = f"""
Please evaluate the following AI assistant response on multiple dimensions:

USER QUERY: {query}

CONTEXT: {context}

AI RESPONSE: {response}
"""

        if expected_response:
            prompt += f"\nEXPECTED RESPONSE: {expected_response}"

        if criteria:
            prompt += f"\nSPECIFIC CRITERIA: {', '.join(criteria)}"

        prompt += """

Please evaluate the response on these dimensions (scale 1-10):

1. RELEVANCE: How well does the response address the user's query?
2. ACCURACY: How factually correct is the information provided?
3. COMPLETENESS: How complete is the response? Does it answer all parts of the question?
4. CLARITY: How clear and easy to understand is the response?
5. HELPFULNESS: How helpful would this response be to the user?
6. OVERALL: Overall quality of the response

For each dimension, provide:
- A score from 1-10 (where 10 is excellent)
- A brief explanation of your reasoning

Format your response as:
RELEVANCE: [score] - [explanation]
ACCURACY: [score] - [explanation]
COMPLETENESS: [score] - [explanation]
CLARITY: [score] - [explanation]
HELPFULNESS: [score] - [explanation]
OVERALL: [score] - [explanation]
"""

        return prompt

    def _parse_evaluation_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse scores from LLM evaluation response"""
        scores = {}

        # Regular expression to extract scores
        score_pattern = r'(\w+):\s*(\d+(?:\.\d+)?)'

        matches = re.findall(score_pattern, evaluation_text, re.IGNORECASE)

        for dimension, score_str in matches:
            try:
                score = float(score_str)
                # Normalize to 0-1 scale if needed
                if score > 1:
                    score = score / 10.0
                scores[dimension.lower()] = score
            except ValueError:
                continue

        return scores

    def _aggregate_response_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate response metrics across all test cases"""
        if not results:
            return {}

        metrics_lists = {
            "relevance_score": [],
            "accuracy_score": [],
            "completeness_score": [],
            "clarity_score": [],
            "helpfulness_score": [],
            "overall_score": []
        }

        for result in results:
            metrics = result["metrics"]
            for key in metrics_lists:
                metrics_lists[key].append(getattr(metrics, key))

        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in metrics_lists.items()
        }

    def _metrics_by_difficulty(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics grouped by difficulty level"""
        by_difficulty = {}

        for difficulty in ["easy", "medium", "hard"]:
            difficulty_results = [r for r in results if r["difficulty"] == difficulty]
            if difficulty_results:
                by_difficulty[difficulty] = self._aggregate_response_metrics(difficulty_results)

        return by_difficulty

    async def evaluate_hallucination(
        self,
        query: str,
        response: str,
        source_documents: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if the response contains hallucinations
        (information not supported by source documents)
        """

        hallucination_prompt = f"""
Analyze the following AI response for potential hallucinations - information that is not supported by the provided source documents.

USER QUERY: {query}

AI RESPONSE: {response}

SOURCE DOCUMENTS:
{chr(10).join([f"Document {i+1}: {doc}" for i, doc in enumerate(source_documents)])}

Please identify:
1. SUPPORTED CLAIMS: Information that is clearly supported by the source documents
2. UNSUPPORTED CLAIMS: Information that cannot be verified from the source documents
3. CONTRADICTORY CLAIMS: Information that contradicts the source documents
4. HALLUCINATION SCORE: Rate from 1-10 (1 = heavy hallucination, 10 = no hallucination)

Format your response as:
SUPPORTED: [list of supported claims]
UNSUPPORTED: [list of unsupported claims]
CONTRADICTORY: [list of contradictory claims]
HALLUCINATION_SCORE: [score]
EXPLANATION: [brief explanation of your assessment]
"""

        try:
            eval_response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert fact-checker specializing in identifying hallucinations in AI responses."
                    },
                    {
                        "role": "user",
                        "content": hallucination_prompt
                    }
                ],
                temperature=0.1
            )

            return {
                "query": query,
                "response": response,
                "evaluation": eval_response.choices[0].message.content,
                "source_documents": source_documents
            }

        except Exception as e:
            return {
                "query": query,
                "response": response,
                "evaluation": f"Error in hallucination evaluation: {e}",
                "source_documents": source_documents
            }

    def print_summary(self, results: Dict[str, Any]):
        """Print a human-readable summary of response evaluation results"""
        metrics = results["aggregate_metrics"]

        print(f"\n🤖 RESPONSE QUALITY EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Test Cases: {results['total_test_cases']}")
        print(f"Agent ID: {results['agent_id']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"\n📊 OVERALL METRICS (0-1 scale):")
        print(f"  Relevance: {metrics['relevance_score']:.3f}")
        print(f"  Accuracy: {metrics['accuracy_score']:.3f}")
        print(f"  Completeness: {metrics['completeness_score']:.3f}")
        print(f"  Clarity: {metrics['clarity_score']:.3f}")
        print(f"  Helpfulness: {metrics['helpfulness_score']:.3f}")
        print(f"  Overall: {metrics['overall_score']:.3f}")

        # Print difficulty breakdown
        if "per_difficulty_metrics" in results:
            print(f"\n📈 BY DIFFICULTY:")
            for difficulty, diff_metrics in results["per_difficulty_metrics"].items():
                print(f"  {difficulty.upper()}:")
                print(f"    Relevance: {diff_metrics['relevance_score']:.3f}")
                print(f"    Accuracy: {diff_metrics['accuracy_score']:.3f}")
                print(f"    Overall: {diff_metrics['overall_score']:.3f}")

    async def generate_test_cases(
        self,
        agent_context: str,
        num_cases: int = 10
    ) -> List[ResponseTestCase]:
        """Generate test cases based on agent context"""

        generation_prompt = f"""
Generate {num_cases} test cases for evaluating an AI assistant's responses.

AGENT CONTEXT: {agent_context}

For each test case, provide:
1. A realistic user query
2. The context/scenario
3. Difficulty level (easy/medium/hard)

Generate diverse queries covering different types of requests (informational, procedural, troubleshooting, etc.).

Format as JSON array:
[
  {{
    "query": "user question",
    "context": "scenario description",
    "difficulty": "easy|medium|hard"
  }}
]
"""

        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating comprehensive test cases for AI systems."
                    },
                    {
                        "role": "user",
                        "content": generation_prompt
                    }
                ],
                temperature=0.7
            )

            test_cases_json = response.choices[0].message.content
            test_cases_data = json.loads(test_cases_json)

            return [
                ResponseTestCase(
                    query=case["query"],
                    context=case["context"],
                    difficulty=case["difficulty"]
                )
                for case in test_cases_data
            ]

        except Exception as e:
            print(f"Error generating test cases: {e}")
            # Return default test cases
            return [
                ResponseTestCase(
                    query="What is your refund policy?",
                    context="Customer wants to return a product",
                    difficulty="easy"
                ),
                ResponseTestCase(
                    query="How can I integrate your API with my existing authentication system?",
                    context="Developer needs technical integration help",
                    difficulty="hard"
                )
            ]


# Example usage
async def main():
    """Example of how to use the response evaluator"""
    print("Response evaluator ready to use!")


if __name__ == "__main__":
    asyncio.run(main())