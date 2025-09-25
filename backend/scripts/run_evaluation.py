#!/usr/bin/env python3
"""
Script to run RAG system evaluation.
Can be used for development, CI/CD, or production monitoring.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.evaluation.retrieval_evaluator import RetrievalEvaluator, RetrievalTestCase
from app.evaluation.response_evaluator import ResponseEvaluator, ResponseTestCase
from app.evaluation.end_to_end_evaluator import EndToEndEvaluator, E2ETestCase
from app.services.rag_service import RAGService
from app.core.database import get_db
from app.core.config import settings


async def run_retrieval_evaluation(agent_id: int, output_file: str = None):
    """Run retrieval evaluation only"""

    print("🔍 Running Retrieval Evaluation...")

    # Initialize RAG service (you'll need to implement proper initialization)
    # For now, we'll create a placeholder
    rag_service = RAGService()  # You'll need to pass proper dependencies
    evaluator = RetrievalEvaluator(rag_service)

    # Example test cases - in practice, these would come from your data
    test_cases = [
        RetrievalTestCase(
            query="What is your refund policy?",
            expected_chunks=["refund-policy-section", "return-process"],
            context="Customer wants to return a product",
            difficulty="easy"
        ),
        RetrievalTestCase(
            query="How do I configure advanced authentication settings?",
            expected_chunks=["auth-config-docs", "security-settings", "api-keys"],
            context="Developer setting up enterprise auth",
            difficulty="hard"
        ),
        RetrievalTestCase(
            query="What are the pricing options?",
            expected_chunks=["pricing-plans", "billing-info"],
            context="Potential customer asking about costs",
            difficulty="medium"
        )
    ]

    # Run evaluation
    results = await evaluator.evaluate_retrieval(test_cases, k=5)

    # Print summary
    evaluator.print_summary(results)

    # Save results if requested
    if output_file:
        evaluator.save_results(results, output_file)
        print(f"📄 Results saved to {output_file}")

    return results


async def run_response_evaluation(agent_id: int, output_file: str = None):
    """Run response quality evaluation only"""

    print("🤖 Running Response Quality Evaluation...")

    # Initialize services
    rag_service = RAGService()
    evaluator = ResponseEvaluator(rag_service)

    # Example test cases
    test_cases = [
        ResponseTestCase(
            query="What is your refund policy?",
            expected_response="Our refund policy allows returns within 30 days...",
            context="Customer service inquiry",
            difficulty="easy"
        ),
        ResponseTestCase(
            query="How do I integrate your API with React?",
            context="Developer needs integration help",
            difficulty="medium"
        ),
        ResponseTestCase(
            query="What are the enterprise security features?",
            context="Enterprise customer evaluation",
            difficulty="hard"
        )
    ]

    # Run evaluation
    results = await evaluator.evaluate_responses(test_cases, agent_id)

    # Print summary
    evaluator.print_summary(results)

    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Results saved to {output_file}")

    return results


async def run_comprehensive_evaluation(
    agent_id: int,
    concurrent_requests: int = 1,
    output_file: str = None
):
    """Run comprehensive end-to-end evaluation"""

    print("🚀 Running Comprehensive Evaluation...")

    # Initialize services
    rag_service = RAGService()
    evaluator = EndToEndEvaluator(rag_service)

    # Example comprehensive test cases
    test_cases = [
        E2ETestCase(
            query="What is your refund policy?",
            expected_chunks=["refund-policy-section"],
            expected_response="Our refund policy allows returns within 30 days of purchase...",
            context="Customer service inquiry",
            difficulty="easy",
            expected_response_time=2.0
        ),
        E2ETestCase(
            query="How do I set up OAuth2 authentication?",
            expected_chunks=["oauth-setup", "auth-configuration"],
            context="Developer integration question",
            difficulty="medium",
            expected_response_time=3.0
        ),
        E2ETestCase(
            query="What are the enterprise-level security and compliance features?",
            expected_chunks=["enterprise-security", "compliance-docs", "audit-features"],
            context="Enterprise evaluation",
            difficulty="hard",
            expected_response_time=4.0
        )
    ]

    # Run comprehensive evaluation
    results = await evaluator.run_comprehensive_evaluation(
        agent_id=agent_id,
        test_cases=test_cases,
        concurrent_requests=concurrent_requests
    )

    # Print summary
    evaluator.print_comprehensive_summary(results)

    # Save results if requested
    if output_file:
        evaluator.save_comprehensive_results(results, output_file)

    return results


async def run_continuous_monitoring(agent_id: int, interval_seconds: int = 300):
    """Run continuous evaluation for production monitoring"""

    print(f"📊 Starting continuous monitoring for agent {agent_id}")
    print(f"⏱️  Evaluation interval: {interval_seconds} seconds")

    iteration = 0

    while True:
        try:
            iteration += 1
            print(f"\n--- Monitoring Iteration {iteration} ---")

            # Run lightweight evaluation
            results = await run_response_evaluation(
                agent_id=agent_id,
                output_file=f"monitoring_results_{iteration}.json"
            )

            # Check for performance degradation
            overall_score = results["aggregate_metrics"]["overall_score"]
            if overall_score < 0.7:  # Threshold for alerting
                print("🚨 ALERT: Performance degradation detected!")
                print(f"   Overall score: {overall_score:.3f} (threshold: 0.70)")
                # Here you could send alerts, notifications, etc.

            print(f"💤 Sleeping for {interval_seconds} seconds...")
            await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying


def main():
    """Main CLI interface"""

    parser = argparse.ArgumentParser(description="RAG System Evaluation Tool")
    parser.add_argument("--agent-id", type=int, default=1, help="Agent ID to evaluate")
    parser.add_argument("--type", choices=["retrieval", "response", "comprehensive", "monitor"],
                       default="comprehensive", help="Type of evaluation to run")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--concurrent", type=int, default=1,
                       help="Number of concurrent requests for load testing")
    parser.add_argument("--monitor-interval", type=int, default=300,
                       help="Interval in seconds for continuous monitoring")

    args = parser.parse_args()

    # Set default output filename if not provided
    if not args.output:
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"evaluation_{args.type}_{timestamp}.json"

    print(f"🎯 Agent ID: {args.agent_id}")
    print(f"📋 Evaluation Type: {args.type}")
    print(f"📄 Output File: {args.output}")

    # Run the appropriate evaluation
    if args.type == "retrieval":
        asyncio.run(run_retrieval_evaluation(args.agent_id, args.output))
    elif args.type == "response":
        asyncio.run(run_response_evaluation(args.agent_id, args.output))
    elif args.type == "comprehensive":
        asyncio.run(run_comprehensive_evaluation(
            args.agent_id, args.concurrent, args.output
        ))
    elif args.type == "monitor":
        asyncio.run(run_continuous_monitoring(args.agent_id, args.monitor_interval))


if __name__ == "__main__":
    main()