#!/usr/bin/env python3
"""
Test script for database service.
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.services.database_service import create_demo_data, db_service


async def test_database():
    """Test database operations"""
    print("🧪 Testing database service...")

    try:
        # Create demo data
        demo_data = await create_demo_data()

        if demo_data:
            # Test retrieving data
            user = demo_data["user"]
            agents = await db_service.get_user_agents(user.id)

            print(f"📊 User: {user.name} ({user.email})")
            print(f"🤖 Agents: {len(agents)}")

            for agent in agents:
                documents = await db_service.get_agent_documents(agent.id)
                stats = await db_service.get_agent_stats(agent.id)
                print(f"   - {agent.name}: {len(documents)} documents, {stats['total_conversations']} conversations")

            # Test system stats
            system_stats = await db_service.get_system_stats()
            print(f"🌍 System stats: {system_stats}")

        print("✅ Database service test completed!")

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_database())