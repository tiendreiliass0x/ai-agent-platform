"""
Simple test for agent creation process
"""

import asyncio
import httpx

async def test_agent_creation():
    async with httpx.AsyncClient() as client:
        # Login
        login_response = await client.post(
            "http://127.0.0.1:8001/api/v1/auth/login",
            data={
                "username": "testuser@example.com",
                "password": "password123"
            }
        )

        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.text}")
            return

        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Get organizations
        orgs_response = await client.get(
            "http://127.0.0.1:8001/api/v1/organizations/",
            headers=headers
        )

        if orgs_response.status_code != 200:
            print(f"❌ Failed to get orgs: {orgs_response.text}")
            return

        orgs = orgs_response.json()
        if not orgs:
            print("❌ No organizations found")
            return

        org_id = orgs[0]["id"]
        print(f"✅ Using organization: {orgs[0]['name']} (ID: {org_id})")

        # Get agent types
        types_response = await client.get("http://127.0.0.1:8001/api/v1/agents/templates/types")
        if types_response.status_code == 200:
            types = types_response.json()
            print(f"✅ Available agent types: {[t['value'] for t in types['agent_types']]}")

        # Get industries
        industries_response = await client.get("http://127.0.0.1:8001/api/v1/agents/templates/industries")
        if industries_response.status_code == 200:
            industries = industries_response.json()
            print(f"✅ Available industries: {[i['value'] for i in industries['industries']]}")

        # Create agent
        agent_data = {
            "name": "Test Support Agent",
            "description": "AI agent for testing",
            "system_prompt": "You are a helpful support agent.",
            "agent_type": "customer_support",
            "industry": "saas",
            "auto_optimize": True,
            "config": {},
            "widget_config": {}
        }

        create_response = await client.post(
            f"http://127.0.0.1:8001/api/v1/agents/?organization_id={org_id}",
            json=agent_data,
            headers=headers
        )

        if create_response.status_code == 200:
            result = create_response.json()
            print("✅ Agent created successfully!")
            print(f"📊 Agent ID: {result['agent']['id']}")
            print(f"🤖 Agent Name: {result['agent']['name']}")
            print(f"⚡ Optimization Applied: {result['optimization_applied']}")
            print(f"📋 Template Used: {result['template_used']}")
            print(f"💡 Recommendations: {len(result['recommendations'])} items")
            print(f"📚 Setup Steps: {len(result['setup_guide']['steps'])} steps")
        else:
            print(f"❌ Agent creation failed: {create_response.status_code}")
            print(f"Error: {create_response.text}")

if __name__ == "__main__":
    asyncio.run(test_agent_creation())