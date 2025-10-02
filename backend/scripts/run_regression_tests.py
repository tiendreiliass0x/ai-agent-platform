#!/usr/bin/env python3
"""
Regression test runner script.
Runs comprehensive tests to prevent schema and integration regressions.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any


class RegressionTestRunner:
    """Runs comprehensive regression tests."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[str, Any] = {}

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=Path(__file__).parent.parent
            )

            if result.returncode == 0:
                self.log(f"✅ {description} - PASSED")
                if self.verbose and result.stdout:
                    print(result.stdout)
                return True
            else:
                self.log(f"❌ {description} - FAILED")
                print(f"STDERR: {result.stderr}")
                if result.stdout:
                    print(f"STDOUT: {result.stdout}")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"⏰ {description} - TIMEOUT")
            return False
        except Exception as e:
            self.log(f"💥 {description} - ERROR: {e}")
            return False

    def run_schema_validation_tests(self) -> bool:
        """Run schema validation tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/test_schema_validation.py",
            "-v", "-x", "--tb=short"
        ]
        return self.run_command(cmd, "Schema Validation Tests")

    def run_migration_tests(self) -> bool:
        """Run migration tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/test_migrations.py",
            "-v", "-x", "--tb=short"
        ]
        return self.run_command(cmd, "Database Migration Tests")

    def run_model_validation_tests(self) -> bool:
        """Run model validation tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/test_models.py",
            "-v", "-x", "--tb=short"
        ]
        return self.run_command(cmd, "Model Validation Tests")

    def run_database_integration_tests(self) -> bool:
        """Run database integration tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/test_database_integration.py",
            "-v", "-x", "--tb=short",
            "-k", "test_agent_public_id"  # Focus on regression-specific tests
        ]
        return self.run_command(cmd, "Database Integration Tests (Regression Focus)")

    def run_service_layer_tests(self) -> bool:
        """Run service layer tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/test_service_layer.py",
            "-v", "-x", "--tb=short"
        ]
        return self.run_command(cmd, "Service Layer Tests")

    def run_api_tests(self) -> bool:
        """Run API endpoint tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/test_api_endpoints.py",
            "-v", "-x", "--tb=short",
            "-k", "test_user_context"  # Focus on the original failing endpoint
        ]
        return self.run_command(cmd, "API Endpoint Tests (User Context)")

    def run_ci_integration_tests(self) -> bool:
        """Run CI/CD integration tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/test_ci_integration.py",
            "-v", "-x", "--tb=short"
        ]
        return self.run_command(cmd, "CI/CD Integration Tests")

    def check_environment(self) -> bool:
        """Check test environment setup."""
        self.log("Checking test environment...")

        # Set testing environment
        os.environ["TESTING"] = "1"
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print("❌ Python 3.8+ required")
            return False

        # Check required packages
        required_packages = [
            "pytest", "pytest-asyncio", "sqlalchemy", "alembic", "fastapi"
        ]

        for package in required_packages:
            try:
                __import__(package)
                self.log(f"✅ {package} available")
            except ImportError:
                print(f"❌ Required package '{package}' not available")
                return False

        self.log("✅ Environment check passed")
        return True

    def run_all_regression_tests(self) -> Dict[str, bool]:
        """Run all regression tests and return results."""
        if not self.check_environment():
            return {"environment": False}

        test_suites = [
            ("Schema Validation", self.run_schema_validation_tests),
            ("Model Validation", self.run_model_validation_tests),
            ("Migration Tests", self.run_migration_tests),
            ("Database Integration", self.run_database_integration_tests),
            ("Service Layer", self.run_service_layer_tests),
            ("API Tests", self.run_api_tests),
            ("CI Integration", self.run_ci_integration_tests),
        ]

        results = {}
        for name, test_func in test_suites:
            try:
                results[name] = test_func()
            except Exception as e:
                self.log(f"💥 {name} failed with exception: {e}")
                results[name] = False

        return results

    def print_summary(self, results: Dict[str, bool]):
        """Print test results summary."""
        print("\n" + "="*60)
        print("REGRESSION TEST SUMMARY")
        print("="*60)

        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        failed_tests = total_tests - passed_tests

        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:<25} {status}")

        print("\n" + "-"*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")

        if failed_tests == 0:
            print("\n🎉 ALL REGRESSION TESTS PASSED!")
            print("The system is protected against the public_id regression.")
        else:
            print(f"\n⚠️  {failed_tests} TEST SUITE(S) FAILED")
            print("Please review and fix failing tests before deployment.")

        return failed_tests == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run regression tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--suite", choices=[
        "schema", "models", "migrations", "database", "services", "api", "ci", "all"
    ], default="all", help="Test suite to run")

    args = parser.parse_args()

    runner = RegressionTestRunner(verbose=args.verbose)

    if args.suite == "all":
        results = runner.run_all_regression_tests()
        success = runner.print_summary(results)
    else:
        # Run specific test suite
        suite_map = {
            "schema": runner.run_schema_validation_tests,
            "models": runner.run_model_validation_tests,
            "migrations": runner.run_migration_tests,
            "database": runner.run_database_integration_tests,
            "services": runner.run_service_layer_tests,
            "api": runner.run_api_tests,
            "ci": runner.run_ci_integration_tests,
        }

        if not runner.check_environment():
            sys.exit(1)

        success = suite_map[args.suite]()
        print(f"\n{args.suite.upper()} Tests: {'✅ PASSED' if success else '❌ FAILED'}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()