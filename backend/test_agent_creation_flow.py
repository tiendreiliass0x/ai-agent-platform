#!/usr/bin/env python3
"""
Comprehensive Test Suite for Agent Creation Process

This script tests the complete end-to-end agent creation flow including:
1. User registration and authentication
2. Organization creation and user association
3. User context endpoint with organization data
4. Agent creation with organization context
5. Verification of the complete flow

Usage: python test_agent_creation_flow.py
"""

import requests
import json
import time
import random
import string
from typing import Dict, Any, Optional

class AgentCreationTestSuite:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_data = {}
        self.access_token = None

    def log(self, message: str, status: str = "INFO"):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{status}] {message}")

    def generate_random_string(self, length: int = 8) -> str:
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None,
                    headers: Optional[Dict] = None, expect_status: int = 200) -> Dict[Any, Any]:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"

        if headers is None:
            headers = {"Content-Type": "application/json"}

        if self.access_token and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.access_token}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers)
            elif method.upper() == "POST":
                if "application/x-www-form-urlencoded" in headers.get("Content-Type", ""):
                    response = self.session.post(url, data=data, headers=headers)
                else:
                    response = self.session.post(url, json=data, headers=headers)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            self.log(f"{method} {endpoint} -> {response.status_code}")

            if response.status_code != expect_status:
                self.log(f"Expected {expect_status}, got {response.status_code}", "ERROR")
                self.log(f"Response: {response.text}", "ERROR")
                return {"error": True, "status_code": response.status_code, "response": response.text}

            try:
                return response.json()
            except ValueError:
                return {"success": True, "text": response.text}

        except Exception as e:
            self.log(f"Request failed: {str(e)}", "ERROR")
            return {"error": True, "exception": str(e)}

    def test_step_1_user_registration(self) -> bool:
        """Test Step 1: User Registration"""
        self.log("=== STEP 1: User Registration ===")

        # Generate unique test user data
        random_id = self.generate_random_string()
        self.test_data["user"] = {
            "email": f"testuser_{random_id}@example.com",
            "password": "testpass123",
            "name": f"Test User {random_id}",
            "plan": "free"
        }

        # Register user
        result = self.make_request(
            "POST",
            "/api/v1/auth/register",
            data=self.test_data["user"]
        )

        if "error" in result:
            self.log("User registration failed", "ERROR")
            return False

        self.test_data["user_id"] = result.get("id")
        self.log(f"✅ User registered successfully with ID: {self.test_data['user_id']}")
        return True

    def test_step_2_user_authentication(self) -> bool:
        """Test Step 2: User Authentication"""
        self.log("=== STEP 2: User Authentication ===")

        # Login user
        login_data = {
            "username": self.test_data["user"]["email"],
            "password": self.test_data["user"]["password"]
        }

        result = self.make_request(
            "POST",
            "/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if "error" in result:
            self.log("User authentication failed", "ERROR")
            return False

        self.access_token = result.get("access_token")
        if not self.access_token:
            self.log("No access token received", "ERROR")
            return False

        self.log("✅ User authenticated successfully")

        # Verify token with /auth/me
        me_result = self.make_request("GET", "/api/v1/auth/me")
        if "error" in me_result:
            self.log("Token verification failed", "ERROR")
            return False

        self.log(f"✅ Token verified - User: {me_result.get('name')}")
        return True

    def test_step_3_organization_creation(self) -> bool:
        """Test Step 3: Organization Creation and User Association"""
        self.log("=== STEP 3: Organization Creation ===")

        # Generate organization data
        random_id = self.generate_random_string()
        self.test_data["organization"] = {
            "name": f"Test Organization {random_id}",
            "slug": f"test-org-{random_id}",
            "description": "Test organization for agent creation",
            "plan": "free"
        }

        # Create organization
        result = self.make_request(
            "POST",
            "/api/v1/organizations/",
            data=self.test_data["organization"]
        )

        if "error" in result:
            self.log("Organization creation failed", "ERROR")
            return False

        self.test_data["organization_id"] = result.get("id")
        self.log(f"✅ Organization created successfully with ID: {self.test_data['organization_id']}")

        # Verify organization was created
        org_result = self.make_request("GET", f"/api/v1/organizations/{self.test_data['organization_id']}")
        if "error" in org_result:
            self.log("Organization verification failed", "ERROR")
            return False

        self.log("✅ Organization verified successfully")
        return True

    def test_step_4_user_context_endpoint(self) -> bool:
        """Test Step 4: User Context Endpoint with Organization Data"""
        self.log("=== STEP 4: User Context Endpoint ===")

        # Test main user context endpoint
        context_result = self.make_request("GET", "/api/v1/users/context")

        if "error" in context_result:
            self.log("User context endpoint failed", "ERROR")
            return False

        # Verify response structure
        required_fields = ["id", "email", "name", "organizations"]
        for field in required_fields:
            if field not in context_result:
                self.log(f"Missing required field in context: {field}", "ERROR")
                return False

        organizations = context_result.get("organizations", [])
        if not organizations:
            self.log("No organizations found in user context", "ERROR")
            return False

        # Find our test organization
        test_org = None
        for org in organizations:
            if org.get("id") == self.test_data["organization_id"]:
                test_org = org
                break

        if not test_org:
            self.log("Test organization not found in user context", "ERROR")
            return False

        # Verify organization context structure
        org_required_fields = ["id", "name", "plan", "max_agents", "can_add_agent", "user_role", "user_permissions"]
        for field in org_required_fields:
            if field not in test_org:
                self.log(f"Missing required field in organization context: {field}", "ERROR")
                return False

        self.log("✅ User context endpoint working correctly")
        self.log(f"   - Organization: {test_org['name']}")
        self.log(f"   - User Role: {test_org['user_role']}")
        self.log(f"   - Can Add Agent: {test_org['can_add_agent']}")

        # Test organizations endpoint
        orgs_result = self.make_request("GET", "/api/v1/users/organizations")
        if "error" in orgs_result:
            self.log("User organizations endpoint failed", "ERROR")
            return False

        self.log("✅ User organizations endpoint working correctly")

        # Test specific organization context endpoint
        org_context_result = self.make_request(
            "GET",
            f"/api/v1/users/organizations/{self.test_data['organization_id']}/context"
        )

        if "error" in org_context_result:
            self.log("Specific organization context endpoint failed", "ERROR")
            return False

        self.log("✅ Specific organization context endpoint working correctly")
        return True

    def test_step_5_agent_creation(self) -> bool:
        """Test Step 5: Agent Creation with Organization Context"""
        self.log("=== STEP 5: Agent Creation ===")

        # Generate agent data
        random_id = self.generate_random_string()
        self.test_data["agent"] = {
            "name": f"Test Agent {random_id}",
            "description": "Test agent created via API",
            "system_prompt": "You are a helpful test assistant.",
            "tier": "basic",
            "domain_expertise_enabled": False,
            "tool_policy": "permissive",
            "grounding_mode": "strict"
        }

        # Create agent with organization context
        result = self.make_request(
            "POST",
            f"/api/v1/agents/?organization_id={self.test_data['organization_id']}",
            data=self.test_data["agent"]
        )

        if "error" in result:
            self.log("Agent creation failed", "ERROR")
            return False

        self.test_data["agent_id"] = result.get("id")
        self.log(f"✅ Agent created successfully with ID: {self.test_data['agent_id']}")

        # Verify agent was created with correct organization
        agent_result = self.make_request("GET", f"/api/v1/agents/{self.test_data['agent_id']}")
        if "error" in agent_result:
            self.log("Agent verification failed", "ERROR")
            return False

        if agent_result.get("organization_id") != self.test_data["organization_id"]:
            self.log("Agent not associated with correct organization", "ERROR")
            return False

        self.log("✅ Agent verified successfully")
        self.log(f"   - Agent: {agent_result['name']}")
        self.log(f"   - Organization ID: {agent_result['organization_id']}")
        return True

    def test_step_6_end_to_end_verification(self) -> bool:
        """Test Step 6: End-to-End Flow Verification"""
        self.log("=== STEP 6: End-to-End Verification ===")

        # Verify the complete flow by checking all created entities

        # 1. User still exists and has correct data
        user_result = self.make_request("GET", "/api/v1/auth/me")
        if "error" in user_result or user_result.get("id") != self.test_data["user_id"]:
            self.log("User verification failed", "ERROR")
            return False

        # 2. Organization still exists and has correct data
        org_result = self.make_request("GET", f"/api/v1/organizations/{self.test_data['organization_id']}")
        if "error" in org_result or org_result.get("id") != self.test_data["organization_id"]:
            self.log("Organization verification failed", "ERROR")
            return False

        # 3. Agent still exists and is associated with organization
        agent_result = self.make_request("GET", f"/api/v1/agents/{self.test_data['agent_id']}")
        if "error" in agent_result or agent_result.get("id") != self.test_data["agent_id"]:
            self.log("Agent verification failed", "ERROR")
            return False

        # 4. User context shows complete relationship
        context_result = self.make_request("GET", "/api/v1/users/context")
        if "error" in context_result:
            self.log("Final context verification failed", "ERROR")
            return False

        # 5. Organization shows agent count
        orgs = context_result.get("organizations", [])
        test_org = next((org for org in orgs if org["id"] == self.test_data["organization_id"]), None)

        if not test_org:
            self.log("Organization not found in final context", "ERROR")
            return False

        agents_count = test_org.get("agents_count", 0)
        if agents_count < 1:
            self.log(f"Expected at least 1 agent, found {agents_count}", "ERROR")
            return False

        self.log("✅ End-to-end verification successful")
        self.log(f"   - User ID: {self.test_data['user_id']}")
        self.log(f"   - Organization ID: {self.test_data['organization_id']}")
        self.log(f"   - Agent ID: {self.test_data['agent_id']}")
        self.log(f"   - Agents in Organization: {agents_count}")
        return True

    def run_complete_test_suite(self) -> bool:
        """Run the complete test suite"""
        self.log("🚀 Starting Agent Creation Test Suite")
        self.log(f"Testing against: {self.base_url}")

        test_steps = [
            ("User Registration", self.test_step_1_user_registration),
            ("User Authentication", self.test_step_2_user_authentication),
            ("Organization Creation", self.test_step_3_organization_creation),
            ("User Context Endpoint", self.test_step_4_user_context_endpoint),
            ("Agent Creation", self.test_step_5_agent_creation),
            ("End-to-End Verification", self.test_step_6_end_to_end_verification)
        ]

        results = {}

        for step_name, test_function in test_steps:
            self.log(f"\n{'='*50}")
            try:
                result = test_function()
                results[step_name] = result

                if result:
                    self.log(f"✅ {step_name} PASSED", "SUCCESS")
                else:
                    self.log(f"❌ {step_name} FAILED", "ERROR")
                    break  # Stop on first failure

            except Exception as e:
                self.log(f"❌ {step_name} FAILED with exception: {str(e)}", "ERROR")
                results[step_name] = False
                break

        # Print final summary
        self.log(f"\n{'='*50}")
        self.log("🏁 TEST SUITE SUMMARY")
        self.log(f"{'='*50}")

        all_passed = True
        for step_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.log(f"{step_name}: {status}")
            if not result:
                all_passed = False

        if all_passed:
            self.log("\n🎉 ALL TESTS PASSED! Agent creation flow is working correctly.", "SUCCESS")
            self.log("\nTest Data Summary:")
            self.log(f"  User: {self.test_data.get('user', {}).get('email')}")
            self.log(f"  Organization: {self.test_data.get('organization', {}).get('name')}")
            self.log(f"  Agent: {self.test_data.get('agent', {}).get('name')}")
        else:
            self.log("\n💥 SOME TESTS FAILED! Check the logs above for details.", "ERROR")

        return all_passed

    def cleanup_test_data(self):
        """Clean up test data (optional)"""
        self.log("\n🧹 Cleanup is not implemented - test data will remain in database")
        self.log("This allows manual verification of the created entities")

def main():
    """Main entry point"""
    print("🧪 Agent Creation Flow Test Suite")
    print("=" * 50)

    # Initialize test suite
    test_suite = AgentCreationTestSuite()

    # Run complete test suite
    success = test_suite.run_complete_test_suite()

    # Optional cleanup
    # test_suite.cleanup_test_data()

    # Exit with appropriate code
    exit(0 if success else 1)

if __name__ == "__main__":
    main()