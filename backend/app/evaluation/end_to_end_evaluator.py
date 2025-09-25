"""
End-to-end evaluation system for the complete RAG pipeline.
Combines retrieval and response evaluation with additional system metrics.
"""

import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics

from .retrieval_evaluator import RetrievalEvaluator, RetrievalTestCase
from .response_evaluator import ResponseEvaluator, ResponseTestCase
from ..services.rag_service import RAGService


@dataclass
class E2ETestCase:
    """End-to-end test case combining retrieval and response evaluation"""
    query: str
    expected_chunks: List[str]
    expected_response: Optional[str] = None
    context: str = ""
    difficulty: str = "medium"
    expected_response_time: float = 5.0  # seconds


@dataclass
class SystemMetrics:
    """System performance metrics"""
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    success_rate: float
    error_rate: float
    throughput: float  # requests per second


class EndToEndEvaluator:
    """Comprehensive evaluation of the entire RAG system"""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.retrieval_evaluator = RetrievalEvaluator(rag_service)
        self.response_evaluator = ResponseEvaluator(rag_service)

    async def run_comprehensive_evaluation(
        self,
        agent_id: int,
        test_cases: List[E2ETestCase],
        concurrent_requests: int = 1
    ) -> Dict[str, Any]:
        """
        Run comprehensive end-to-end evaluation

        Args:
            agent_id: Agent to evaluate
            test_cases: List of test cases
            concurrent_requests: Number of concurrent requests for load testing

        Returns:
            Complete evaluation results
        """
        print(f"🚀 Starting comprehensive evaluation for agent {agent_id}")
        print(f"📊 Test cases: {len(test_cases)}")
        print(f"⚡ Concurrent requests: {concurrent_requests}")

        start_time = time.time()

        # 1. System Performance Evaluation
        print("\n1️⃣ Evaluating system performance...")
        system_metrics = await self._evaluate_system_performance(
            agent_id, test_cases, concurrent_requests
        )

        # 2. Retrieval Evaluation
        print("\n2️⃣ Evaluating retrieval accuracy...")
        retrieval_test_cases = [
            RetrievalTestCase(
                query=tc.query,
                expected_chunks=tc.expected_chunks,
                context=tc.context,
                difficulty=tc.difficulty
            )
            for tc in test_cases
        ]
        retrieval_results = await self.retrieval_evaluator.evaluate_retrieval(
            retrieval_test_cases
        )

        # 3. Response Quality Evaluation
        print("\n3️⃣ Evaluating response quality...")
        response_test_cases = [
            ResponseTestCase(
                query=tc.query,
                expected_response=tc.expected_response,
                context=tc.context,
                difficulty=tc.difficulty
            )
            for tc in test_cases
        ]
        response_results = await self.response_evaluator.evaluate_responses(
            response_test_cases, agent_id
        )

        # 4. Hallucination Detection
        print("\n4️⃣ Checking for hallucinations...")
        hallucination_results = await self._evaluate_hallucinations(
            agent_id, test_cases[:5]  # Sample for hallucination check
        )

        # 5. User Experience Metrics
        print("\n5️⃣ Calculating user experience metrics...")
        ux_metrics = self._calculate_ux_metrics(
            system_metrics, retrieval_results, response_results
        )

        total_time = time.time() - start_time

        comprehensive_results = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "evaluation_time_seconds": total_time,
            "test_summary": {
                "total_test_cases": len(test_cases),
                "concurrent_requests": concurrent_requests,
                "evaluation_duration": f"{total_time:.2f}s"
            },
            "system_metrics": system_metrics,
            "retrieval_evaluation": retrieval_results,
            "response_evaluation": response_results,
            "hallucination_check": hallucination_results,
            "user_experience_score": ux_metrics,
            "recommendations": self._generate_recommendations(
                system_metrics, retrieval_results, response_results, ux_metrics
            )
        }

        print(f"\n✅ Evaluation completed in {total_time:.2f} seconds")
        return comprehensive_results

    async def _evaluate_system_performance(
        self,
        agent_id: int,
        test_cases: List[E2ETestCase],
        concurrent_requests: int
    ) -> SystemMetrics:
        """Evaluate system performance metrics"""

        response_times = []
        errors = 0
        start_time = time.time()

        # Run test cases with concurrency
        if concurrent_requests > 1:
            # Concurrent execution
            tasks = []
            for test_case in test_cases:
                for _ in range(concurrent_requests):
                    task = self._measure_single_request(agent_id, test_case.query)
                    tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                else:
                    response_times.append(result)
        else:
            # Sequential execution
            for test_case in test_cases:
                try:
                    response_time = await self._measure_single_request(
                        agent_id, test_case.query
                    )
                    response_times.append(response_time)
                except Exception:
                    errors += 1

        total_time = time.time() - start_time
        total_requests = len(test_cases) * concurrent_requests

        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            avg_response_time = p95_response_time = p99_response_time = 0.0

        success_rate = (total_requests - errors) / total_requests if total_requests > 0 else 0.0
        error_rate = errors / total_requests if total_requests > 0 else 0.0
        throughput = total_requests / total_time if total_time > 0 else 0.0

        return SystemMetrics(
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            success_rate=success_rate,
            error_rate=error_rate,
            throughput=throughput
        )

    async def _measure_single_request(self, agent_id: int, query: str) -> float:
        """Measure response time for a single request"""
        start_time = time.time()

        try:
            await self.rag_service.generate_response(
                query=query,
                agent_id=agent_id,
                conversation_id=None
            )
            return time.time() - start_time
        except Exception as e:
            raise Exception(f"Request failed: {e}")

    async def _evaluate_hallucinations(
        self,
        agent_id: int,
        test_cases: List[E2ETestCase]
    ) -> Dict[str, Any]:
        """Check for hallucinations in responses"""

        hallucination_results = []

        for test_case in test_cases:
            # Get response and source documents
            response = await self.rag_service.generate_response(
                query=test_case.query,
                agent_id=agent_id,
                conversation_id=None
            )

            # Get source documents used for this response
            source_chunks = await self.rag_service.retrieve_context(
                query=test_case.query,
                agent_id=agent_id,
                top_k=5
            )

            # Evaluate hallucination
            hallucination_eval = await self.response_evaluator.evaluate_hallucination(
                query=test_case.query,
                response=response,
                source_documents=source_chunks
            )

            hallucination_results.append(hallucination_eval)

        return {
            "total_checked": len(hallucination_results),
            "results": hallucination_results
        }

    def _calculate_ux_metrics(
        self,
        system_metrics: SystemMetrics,
        retrieval_results: Dict[str, Any],
        response_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate user experience score based on all metrics"""

        # Weight different aspects of user experience
        weights = {
            "response_time": 0.25,    # Fast responses
            "accuracy": 0.30,         # Accurate information
            "helpfulness": 0.25,      # Helpful responses
            "reliability": 0.20       # System reliability
        }

        # Normalize response time (assume 3s is good, 1s is excellent)
        response_time_score = max(0, 1 - (system_metrics.avg_response_time - 1) / 2)

        # Get accuracy from retrieval and response metrics
        retrieval_accuracy = retrieval_results["aggregate_metrics"]["f1_at_k"]
        response_accuracy = response_results["aggregate_metrics"]["accuracy_score"]
        accuracy_score = (retrieval_accuracy + response_accuracy) / 2

        # Helpfulness from response metrics
        helpfulness_score = response_results["aggregate_metrics"]["helpfulness_score"]

        # Reliability from system success rate
        reliability_score = system_metrics.success_rate

        # Calculate weighted UX score
        ux_score = (
            weights["response_time"] * response_time_score +
            weights["accuracy"] * accuracy_score +
            weights["helpfulness"] * helpfulness_score +
            weights["reliability"] * reliability_score
        )

        return {
            "overall_ux_score": ux_score,
            "response_time_score": response_time_score,
            "accuracy_score": accuracy_score,
            "helpfulness_score": helpfulness_score,
            "reliability_score": reliability_score
        }

    def _generate_recommendations(
        self,
        system_metrics: SystemMetrics,
        retrieval_results: Dict[str, Any],
        response_results: Dict[str, Any],
        ux_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations based on evaluation results"""

        recommendations = []

        # Performance recommendations
        if system_metrics.avg_response_time > 3.0:
            recommendations.append(
                "⚡ Response time is slow (>3s). Consider optimizing embeddings or using faster models."
            )

        if system_metrics.error_rate > 0.05:
            recommendations.append(
                "🚨 High error rate (>5%). Check system stability and error handling."
            )

        # Retrieval recommendations
        retrieval_f1 = retrieval_results["aggregate_metrics"]["f1_at_k"]
        if retrieval_f1 < 0.7:
            recommendations.append(
                "🎯 Low retrieval accuracy (<70%). Consider improving document chunking or embeddings."
            )

        # Response quality recommendations
        response_accuracy = response_results["aggregate_metrics"]["accuracy_score"]
        if response_accuracy < 0.7:
            recommendations.append(
                "📝 Low response accuracy (<70%). Review system prompts or use better language models."
            )

        response_helpfulness = response_results["aggregate_metrics"]["helpfulness_score"]
        if response_helpfulness < 0.7:
            recommendations.append(
                "🤝 Low helpfulness score (<70%). Improve response templates and add more context."
            )

        # Overall UX recommendations
        if ux_metrics["overall_ux_score"] < 0.8:
            recommendations.append(
                "👥 Overall user experience needs improvement. Focus on top-scoring weak areas."
            )

        if not recommendations:
            recommendations.append("🎉 System is performing well! Consider A/B testing new features.")

        return recommendations

    def print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print detailed summary of comprehensive evaluation"""

        print(f"\n🔍 COMPREHENSIVE RAG EVALUATION REPORT")
        print(f"{'='*60}")
        print(f"Agent ID: {results['agent_id']}")
        print(f"Evaluation Time: {results['evaluation_time_seconds']:.2f}s")
        print(f"Test Cases: {results['test_summary']['total_test_cases']}")

        # System Performance
        sys_metrics = results["system_metrics"]
        print(f"\n⚡ SYSTEM PERFORMANCE:")
        print(f"  Avg Response Time: {sys_metrics.avg_response_time:.2f}s")
        print(f"  95th Percentile: {sys_metrics.p95_response_time:.2f}s")
        print(f"  Success Rate: {sys_metrics.success_rate:.1%}")
        print(f"  Throughput: {sys_metrics.throughput:.2f} req/s")

        # Retrieval Performance
        ret_metrics = results["retrieval_evaluation"]["aggregate_metrics"]
        print(f"\n🎯 RETRIEVAL PERFORMANCE:")
        print(f"  Precision@5: {ret_metrics['precision_at_k']:.3f}")
        print(f"  Recall@5: {ret_metrics['recall_at_k']:.3f}")
        print(f"  F1@5: {ret_metrics['f1_at_k']:.3f}")

        # Response Quality
        resp_metrics = results["response_evaluation"]["aggregate_metrics"]
        print(f"\n🤖 RESPONSE QUALITY:")
        print(f"  Accuracy: {resp_metrics['accuracy_score']:.3f}")
        print(f"  Helpfulness: {resp_metrics['helpfulness_score']:.3f}")
        print(f"  Overall: {resp_metrics['overall_score']:.3f}")

        # User Experience
        ux_score = results["user_experience_score"]["overall_ux_score"]
        print(f"\n👥 USER EXPERIENCE SCORE: {ux_score:.3f}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in results["recommendations"]:
            print(f"  {rec}")

    def save_comprehensive_results(self, results: Dict[str, Any], filename: str):
        """Save comprehensive evaluation results"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📄 Results saved to {filename}")


# Example usage
async def main():
    """Example comprehensive evaluation"""

    # Example test cases
    test_cases = [
        E2ETestCase(
            query="What is the refund policy?",
            expected_chunks=["refund-policy-chunk"],
            context="Customer service inquiry",
            difficulty="easy"
        ),
        E2ETestCase(
            query="How do I integrate the API?",
            expected_chunks=["api-docs-chunk", "integration-guide"],
            context="Developer question",
            difficulty="medium"
        )
    ]

    print("End-to-end evaluator ready!")
    print(f"Example with {len(test_cases)} test cases")


if __name__ == "__main__":
    asyncio.run(main())