#!/usr/bin/env python3
"""
Test core functionality improvements without external dependencies.
"""

def test_core_improvements():
    """Test our core improvements"""
    try:
        # Test validation system
        from app.core.validation import AgentValidator, UserValidator, ValidationException
        
        # Test agent validation
        try:
            agent_data = {
                "name": "Test Agent",
                "description": "A test agent",
                "system_prompt": "You are a helpful assistant"
            }
            validated = AgentValidator(**agent_data)
            print("✅ Agent validation works correctly")
        except Exception as e:
            print(f"❌ Agent validation failed: {e}")
            return False
        
        # Test user validation
        try:
            user_data = {
                "email": "test@gmail.com",
                "password": "SecurePass123!",
                "name": "Test User"
            }
            validated = UserValidator(**user_data)
            print("✅ User validation works correctly")
        except Exception as e:
            print(f"❌ User validation failed: {e}")
            return False
        
        # Test response models
        from app.core.responses import StandardResponse, success_response, error_response
        
        success_resp = success_response({"test": "data"}, "Success message")
        assert success_resp.status == "success"
        assert success_resp.data == {"test": "data"}
        print("✅ Response models work correctly")
        
        # Test exception system
        from app.core.exceptions import NotFoundException, ValidationException
        
        try:
            raise NotFoundException("Test not found", "test_resource", 123)
        except NotFoundException as e:
            assert e.message == "Test not found"
            assert e.code == "NOT_FOUND"
            print("✅ Exception system works correctly")
        
        # Test agent service
        from app.services.agent_service import AgentService
        print("✅ Agent service imported successfully")
        
        print("\n🎉 All core functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_core_improvements()
    exit(0 if success else 1)
