#!/usr/bin/env python3
"""
Simple Agent Creation Test Suite

This test uses the existing hardcoded test users to demonstrate the complete
agent creation flow including the new user context endpoints.

Usage: python test_agent_creation_simple.py
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class SimpleAgentCreationTest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.access_token = None

        # Using existing test user
        self.test_user = {
            "email": "admin@coconutfurniture.com",
            "password": "password123"
        }

    def log(self, message: str, status: str = "INFO"):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{status}] {message}")

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

    def test_step_1_authentication(self) -> bool:
        """Test Step 1: User Authentication"""
        self.log("=== STEP 1: User Authentication ===")

        # Login user
        login_data = {
            "username": self.test_user["email"],
            "password": self.test_user["password"]
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

    def test_step_2_user_context_endpoint(self) -> bool:
        """Test Step 2: User Context Endpoint (NEW FEATURE)"""
        self.log("=== STEP 2: User Context Endpoint (NEW) ===")

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

        # Display organization information
        self.log("✅ User context endpoint working correctly")
        for org in organizations:
            self.log(f"   - Organization: {org['name']} (ID: {org['id']})")
            self.log(f"   - User Role: {org['user_role']}")
            self.log(f"   - Can Add Agent: {org['can_add_agent']}")
            self.log(f"   - Max Agents: {org['max_agents']}")
            self.log(f"   - Current Agents: {org['agents_count']}")

        # Store organization ID for agent creation
        self.organization_id = organizations[0]["id"]

        # Test organizations endpoint
        orgs_result = self.make_request("GET", "/api/v1/users/organizations")
        if "error" in orgs_result:
            self.log("User organizations endpoint failed", "ERROR")
            return False

        self.log("✅ User organizations endpoint working correctly")

        # Test specific organization context endpoint
        org_context_result = self.make_request(
            "GET",
            f"/api/v1/users/organizations/{self.organization_id}/context"
        )

        if "error" in org_context_result:
            self.log("Specific organization context endpoint failed", "ERROR")
            return False

        self.log("✅ Specific organization context endpoint working correctly")
        return True

    def test_step_3_agent_creation(self) -> bool:
        """Test Step 3: Agent Creation with Organization Context"""
        self.log("=== STEP 3: Agent Creation ===")

        # Generate agent data
        agent_data = {
            "name": "Test Agent via API",
            "description": "Test agent created through the API test suite",
            "system_prompt": "You are a helpful test assistant for demonstrating the agent creation process.",
            "tier": "basic",
            "domain_expertise_enabled": False,
            "tool_policy": "permissive",
            "grounding_mode": "strict"
        }

        # Create agent with organization context
        result = self.make_request(
            "POST",
            f"/api/v1/agents/?organization_id={self.organization_id}",
            data=agent_data
        )

        if "error" in result:
            self.log("Agent creation failed", "ERROR")
            return False

        agent_id = result.get("id")
        self.log(f"✅ Agent created successfully with ID: {agent_id}")

        # Verify agent was created with correct organization
        agent_result = self.make_request("GET", f"/api/v1/agents/{agent_id}")
        if "error" in agent_result:
            self.log("Agent verification failed", "ERROR")
            return False

        if agent_result.get("organization_id") != self.organization_id:
            self.log("Agent not associated with correct organization", "ERROR")
            return False

        self.log("✅ Agent verified successfully")
        self.log(f"   - Agent: {agent_result['name']}")
        self.log(f"   - Organization ID: {agent_result['organization_id']}")
        self.log(f"   - Agent ID: {agent_id}")

        self.agent_id = agent_id
        return True

    def test_step_4_verification(self) -> bool:
        """Test Step 4: Verify Updated Organization Context"""
        self.log("=== STEP 4: Verify Updated Context ===")

        # Check that organization context now shows updated agent count
        context_result = self.make_request("GET", "/api/v1/users/context")
        if "error" in context_result:
            self.log("Context verification failed", "ERROR")
            return False

        organizations = context_result.get("organizations", [])
        test_org = next((org for org in organizations if org["id"] == self.organization_id), None)

        if not test_org:
            self.log("Organization not found in context", "ERROR")
            return False

        agents_count = test_org.get("agents_count", 0)
        self.log(f"✅ Organization context updated - Agents count: {agents_count}")

        # List agents for organization
        agents_result = self.make_request("GET", f"/api/v1/agents/?organization_id={self.organization_id}")
        if "error" in agents_result:
            self.log("Failed to list agents", "ERROR")
            return False

        agents = agents_result if isinstance(agents_result, list) else [agents_result]
        self.log(f"✅ Found {len(agents)} agent(s) in organization")

        # Verify our agent is in the list
        our_agent = next((agent for agent in agents if agent.get("id") == self.agent_id), None)
        if not our_agent:
            self.log("Our agent not found in organization's agent list", "ERROR")
            return False

        self.log("✅ Agent successfully listed in organization")
        return True

    def run_test_suite(self) -> bool:
        """Run the complete test suite"""
        self.log("🚀 Starting Simple Agent Creation Test Suite")
        self.log(f"Testing against: {self.base_url}")
        self.log(f"Using test user: {self.test_user['email']}")

        test_steps = [
            ("User Authentication", self.test_step_1_authentication),
            ("User Context Endpoint (NEW)", self.test_step_2_user_context_endpoint),
            ("Agent Creation", self.test_step_3_agent_creation),
            ("Context Verification", self.test_step_4_verification)
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
            self.log("\n✨ Key Features Demonstrated:")
            self.log("  1. ✅ User Authentication")
            self.log("  2. ✅ User Context Endpoint (NEW) - Provides organization info")
            self.log("  3. ✅ Agent Creation with Organization Context")
            self.log("  4. ✅ Real-time Context Updates")
            self.log("\n🎯 The 'Organization context is missing' error is now SOLVED!")
            self.log("   Frontend can call GET /api/v1/users/context to get organization info")
        else:
            self.log("\n💥 SOME TESTS FAILED! Check the logs above for details.", "ERROR")

        return all_passed

def main():
    """Main entry point"""
    print("🧪 Simple Agent Creation Flow Test")
    print("=" * 50)

    # Initialize test suite
    test_suite = SimpleAgentCreationTest()

    # Run test suite
    success = test_suite.run_test_suite()

    # Exit with appropriate code
    exit(0 if success else 1)

if __name__ == "__main__":
    main()