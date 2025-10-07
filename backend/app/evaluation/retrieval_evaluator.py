"""
Retrieval evaluation system for RAG pipeline.
Measures retrieval accuracy, relevance, and coverage.
"""

import json
import asyncio
import math
from statistics import fmean
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import openai
from datetime import datetime

from ..services.rag_service import RAGService
from ..core.config import settings


def _mean(values: List[float]) -> float:
    return fmean(values) if values else 0.0


@dataclass
class RetrievalTestCase:
    """Test case for retrieval evaluation"""
    query: str
    expected_chunks: List[str]  # Ground truth relevant chunks
    context: str  # Background context for the query
    difficulty: str  # easy, medium, hard


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation"""
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    ndcg_at_k: float  # Normalized Discounted Cumulative Gain
    mrr: float  # Mean Reciprocal Rank
    hit_rate: float


class RetrievalEvaluator:
    """Evaluates retrieval quality in RAG system"""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    async def evaluate_retrieval(
        self,
        test_cases: List[RetrievalTestCase],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance on test cases

        Args:
            test_cases: List of test cases with queries and expected results
            k: Number of top chunks to retrieve

        Returns:
            Dictionary with evaluation metrics and detailed results
        """
        results = []

        for test_case in test_cases:
            # Retrieve chunks for the query
            retrieved_chunks = await self.rag_service.retrieve_context(
                query=test_case.query,
                agent_id=1,  # Use default agent for testing
                top_k=k
            )

            # Calculate metrics for this test case
            metrics = self._calculate_retrieval_metrics(
                retrieved_chunks,
                test_case.expected_chunks,
                k
            )

            result = {
                "query": test_case.query,
                "difficulty": test_case.difficulty,
                "retrieved_chunks": retrieved_chunks,
                "expected_chunks": test_case.expected_chunks,
                "metrics": metrics
            }
            results.append(result)

        # Aggregate metrics across all test cases
        aggregate_metrics = self._aggregate_metrics(results)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(test_cases),
            "k": k,
            "aggregate_metrics": aggregate_metrics,
            "per_difficulty_metrics": self._metrics_by_difficulty(results),
            "detailed_results": results
        }

    def _calculate_retrieval_metrics(
        self,
        retrieved: List[str],
        expected: List[str],
        k: int
    ) -> RetrievalMetrics:
        """Calculate retrieval metrics for a single query"""

        # Convert to sets for easier comparison
        retrieved_set = set(retrieved[:k])
        expected_set = set(expected)

        # True positives: relevant chunks that were retrieved
        tp = len(retrieved_set.intersection(expected_set))

        # Precision@k: fraction of retrieved chunks that are relevant
        precision = tp / k if k > 0 else 0

        # Recall@k: fraction of relevant chunks that were retrieved
        recall = tp / len(expected_set) if len(expected_set) > 0 else 0

        # F1@k: harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # NDCG@k: Normalized Discounted Cumulative Gain
        ndcg = self._calculate_ndcg(retrieved[:k], expected_set)

        # MRR: Mean Reciprocal Rank
        mrr = self._calculate_mrr(retrieved[:k], expected_set)

        # Hit Rate: 1 if any relevant chunk was retrieved, 0 otherwise
        hit_rate = 1.0 if tp > 0 else 0.0

        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            f1_at_k=f1,
            ndcg_at_k=ndcg,
            mrr=mrr,
            hit_rate=hit_rate
        )

    def _calculate_ndcg(self, retrieved: List[str], expected_set: set) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not retrieved or not expected_set:
            return 0.0

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, chunk in enumerate(retrieved):
            relevance = 1.0 if chunk in expected_set else 0.0
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate IDCG (Ideal DCG) - if all expected chunks were at the top
        idcg = 0.0
        for i in range(min(len(retrieved), len(expected_set))):
            idcg += 1.0 / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_mrr(self, retrieved: List[str], expected_set: set) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, chunk in enumerate(retrieved):
            if chunk in expected_set:
                return 1.0 / (i + 1)
        return 0.0

    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across all test cases"""
        if not results:
            return {}

        metrics_lists = {
            "precision_at_k": [],
            "recall_at_k": [],
            "f1_at_k": [],
            "ndcg_at_k": [],
            "mrr": [],
            "hit_rate": []
        }

        for result in results:
            metrics = result["metrics"]
            for key in metrics_lists:
                metrics_lists[key].append(getattr(metrics, key))

        return {key: _mean(values) for key, values in metrics_lists.items()}

    def _metrics_by_difficulty(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics grouped by difficulty level"""
        by_difficulty = {}

        for difficulty in ["easy", "medium", "hard"]:
            difficulty_results = [r for r in results if r["difficulty"] == difficulty]
            if difficulty_results:
                by_difficulty[difficulty] = self._aggregate_metrics(difficulty_results)

        return by_difficulty

    async def generate_test_cases_from_docs(
        self,
        agent_id: int,
        num_cases: int = 20
    ) -> List[RetrievalTestCase]:
        """
        Auto-generate test cases from existing documents using LLM
        """
        # Get all document chunks for the agent
        # This would need to be implemented based on your document storage
        # For now, return example test cases

        return [
            RetrievalTestCase(
                query="What is the refund policy?",
                expected_chunks=["refund-policy-chunk-1", "refund-policy-chunk-2"],
                context="Customer wants to know about returning items",
                difficulty="easy"
            ),
            RetrievalTestCase(
                query="How do I integrate the API with my React application?",
                expected_chunks=["api-integration-react", "authentication-setup"],
                context="Developer wants to integrate the service",
                difficulty="medium"
            ),
            RetrievalTestCase(
                query="What are the advanced configuration options for enterprise customers?",
                expected_chunks=["enterprise-config", "advanced-settings", "custom-deployment"],
                context="Enterprise customer needs advanced setup",
                difficulty="hard"
            )
        ]

    async def run_benchmark(self, agent_id: int) -> Dict[str, Any]:
        """Run full retrieval benchmark for an agent"""

        # Generate test cases
        test_cases = await self.generate_test_cases_from_docs(agent_id)

        # Run evaluation
        results = await self.evaluate_retrieval(test_cases)

        # Add benchmark metadata
        results["agent_id"] = agent_id
        results["benchmark_type"] = "retrieval_accuracy"

        return results

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def print_summary(self, results: Dict[str, Any]):
        """Print a human-readable summary of results"""
        metrics = results["aggregate_metrics"]

        print(f"\n🔍 RETRIEVAL EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Test Cases: {results['total_test_cases']}")
        print(f"Top-K: {results['k']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"\n📊 OVERALL METRICS:")
        print(f"  Precision@{results['k']}: {metrics['precision_at_k']:.3f}")
        print(f"  Recall@{results['k']}: {metrics['recall_at_k']:.3f}")
        print(f"  F1@{results['k']}: {metrics['f1_at_k']:.3f}")
        print(f"  NDCG@{results['k']}: {metrics['ndcg_at_k']:.3f}")
        print(f"  MRR: {metrics['mrr']:.3f}")
        print(f"  Hit Rate: {metrics['hit_rate']:.3f}")

        # Print difficulty breakdown
        if "per_difficulty_metrics" in results:
            print(f"\n📈 BY DIFFICULTY:")
            for difficulty, diff_metrics in results["per_difficulty_metrics"].items():
                print(f"  {difficulty.upper()}:")
                print(f"    Precision: {diff_metrics['precision_at_k']:.3f}")
                print(f"    Recall: {diff_metrics['recall_at_k']:.3f}")
                print(f"    F1: {diff_metrics['f1_at_k']:.3f}")


# Example usage and test runner
async def main():
    """Example of how to use the retrieval evaluator"""

    # Initialize RAG service (you'd need to pass proper dependencies)
    # rag_service = RAGService(...)
    # evaluator = RetrievalEvaluator(rag_service)

    # # Run benchmark
    # results = await evaluator.run_benchmark(agent_id=1)

    # # Print summary
    # evaluator.print_summary(results)

    # # Save detailed results
    # evaluator.save_results(results, "retrieval_benchmark.json")

    print("Retrieval evaluator ready to use!")


if __name__ == "__main__":
    asyncio.run(main())
