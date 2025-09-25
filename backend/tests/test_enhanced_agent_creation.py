"""
Test script for enhanced agent creation system.
Tests templates, AI optimization, and advanced features.
"""

import asyncio
import json
import httpx
from datetime import datetime

# Test configuration
BASE_URL = "http://127.0.0.1:8001"

async def test_enhanced_agent_creation():
    """Test the enhanced agent creation system"""

    async with httpx.AsyncClient() as client:
        print("🚀 Testing Enhanced Agent Creation System")
        print("=" * 50)

        # Test 1: Login first to get token
        print("\n1. Logging in...")
        login_response = await client.post(
            f"{BASE_URL}/api/v1/auth/login",
            data={
                "username": "testuser@example.com",
                "password": "password123"
            }
        )

        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.status_code} - {login_response.text}")
            return

        token_data = login_response.json()
        token = token_data["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Login successful")

        # Test 1.5: Create an organization for testing
        print("\n1.5. Creating organization...")
        org_data = {
            "name": "Test Tech Company",
            "slug": "test-tech-co",
            "description": "Test organization for agent creation",
            "plan": "pro"
        }
        org_response = await client.post(
            f"{BASE_URL}/api/v1/organizations/",
            json=org_data,
            headers=headers
        )

        if org_response.status_code == 200:
            org_result = org_response.json()
            org_id = org_result["id"]
            print(f"✅ Organization created: {org_result['name']} (ID: {org_id})")
        else:
            print(f"❌ Organization creation failed: {org_response.text}")
            return

        # Test 2: Get available template types
        print("\n2. Getting Agent Types...")
        types_response = await client.get(f"{BASE_URL}/api/v1/agents/templates/types")
        if types_response.status_code == 200:
            types_data = types_response.json()
            print("✅ Available Agent Types:")
            for agent_type in types_data["agent_types"]:
                print(f"   - {agent_type['label']} ({agent_type['value']})")
        else:
            print(f"❌ Failed to get agent types: {types_response.text}")

        # Test 3: Get available industries
        print("\n3. Getting Industries...")
        industries_response = await client.get(f"{BASE_URL}/api/v1/agents/templates/industries")
        if industries_response.status_code == 200:
            industries_data = industries_response.json()
            print("✅ Available Industries:")
            for industry in industries_data["industries"]:
                print(f"   - {industry['label']} ({industry['value']})")
        else:
            print(f"❌ Failed to get industries: {industries_response.text}")

        # Test 4: Create agent with Customer Support template + FinTech industry
        print("\n4. Creating Customer Support Agent for FinTech...")
        agent_data = {
            "name": "FinTech Support Agent",
            "description": "AI agent for financial services customer support",
            "system_prompt": "Help customers with their financial questions and issues.",
            "agent_type": "customer_support",
            "industry": "fintech",
            "auto_optimize": True,
            "config": {},
            "widget_config": {}
        }

        create_response = await client.post(
            f"{BASE_URL}/api/v1/agents/?organization_id={org_id}",
            json=agent_data,
            headers=headers
        )

        if create_response.status_code == 200:
            creation_result = create_response.json()
            print("✅ Agent created successfully!")
            print(f"📊 Agent ID: {creation_result['agent']['id']}")
            print(f"🤖 Agent Name: {creation_result['agent']['name']}")
            print(f"⚡ Optimization Applied: {creation_result['optimization_applied']}")
            print(f"📋 Template Used: {creation_result['template_used']}")

            # Show optimized system prompt (first 200 chars)
            system_prompt = creation_result['agent']['system_prompt']
            print(f"📝 Optimized System Prompt: {system_prompt[:200]}...")

            # Show embed code (first 100 chars)
            embed_code = creation_result['embed_code']
            print(f"🔗 Embed Code: {embed_code[:100]}...")

            # Show setup guide
            setup_guide = creation_result['setup_guide']
            print(f"📚 Setup Steps: {len(setup_guide['steps'])} steps")
            for i, step in enumerate(setup_guide['steps'], 1):
                print(f"   {i}. {step['title']} ({step['estimated_time']})")

            # Show recommendations
            recommendations = creation_result['recommendations']
            print(f"💡 Recommendations: {len(recommendations)} items")
            for rec in recommendations:
                print(f"   - {rec['title']} (Priority: {rec['priority']})")

        else:
            print(f"❌ Agent creation failed: {create_response.status_code}")
            print(f"Error details: {create_response.text}")

        # Test 5: Create agent with Sales Assistant template + SaaS industry
        print("\n5. Creating Sales Assistant Agent for SaaS...")
        sales_agent_data = {
            "name": "SaaS Sales Assistant",
            "description": "AI agent for SaaS sales and lead qualification",
            "system_prompt": "Help qualify leads and assist with SaaS sales inquiries.",
            "agent_type": "sales_assistant",
            "industry": "saas",
            "auto_optimize": True,
            "config": {},
            "widget_config": {}
        }

        sales_response = await client.post(
            f"{BASE_URL}/api/v1/agents/?organization_id={org_id}",
            json=sales_agent_data,
            headers=headers
        )

        if sales_response.status_code == 200:
            sales_result = sales_response.json()
            print("✅ Sales agent created successfully!")
            print(f"📊 Agent ID: {sales_result['agent']['id']}")
            print(f"🤖 Agent Name: {sales_result['agent']['name']}")
            print(f"⚡ Optimization Applied: {sales_result['optimization_applied']}")
            print(f"📋 Template Used: {sales_result['template_used']}")

            # Show config differences
            config = sales_result['agent']['config']
            print(f"⚙️ Agent Config:")
            print(f"   - Temperature: {config.get('temperature', 'N/A')}")
            print(f"   - Lead Scoring: {config.get('lead_scoring', 'N/A')}")
            print(f"   - Follow-up Enabled: {config.get('follow_up_enabled', 'N/A')}")

        else:
            print(f"❌ Sales agent creation failed: {sales_response.status_code}")
            print(f"Error details: {sales_response.text}")

        # Test 6: Create custom agent without template
        print("\n6. Creating Custom Agent...")
        custom_agent_data = {
            "name": "Custom Business Agent",
            "description": "Custom AI agent for specific business needs",
            "system_prompt": "You are a helpful business assistant.",
            "agent_type": "custom",
            "industry": "general",
            "auto_optimize": False,  # No AI optimization
            "config": {"temperature": 0.8},
            "widget_config": {"theme": "custom"}
        }

        custom_response = await client.post(
            f"{BASE_URL}/api/v1/agents/?organization_id={org_id}",
            json=custom_agent_data,
            headers=headers
        )

        if custom_response.status_code == 200:
            custom_result = custom_response.json()
            print("✅ Custom agent created successfully!")
            print(f"📊 Agent ID: {custom_result['agent']['id']}")
            print(f"🤖 Agent Name: {custom_result['agent']['name']}")
            print(f"⚡ Optimization Applied: {custom_result['optimization_applied']}")
            print(f"📋 Template Used: {custom_result['template_used']}")

        else:
            print(f"❌ Custom agent creation failed: {custom_response.status_code}")
            print(f"Error details: {custom_response.text}")

        print("\n" + "=" * 50)
        print("✅ Enhanced Agent Creation System Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_agent_creation())