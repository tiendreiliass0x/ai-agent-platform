# Query Router Setup Guide

## Overview

The Query Router intelligently routes user queries to either:
- **RAG Service** (Phase 1+2): Information retrieval and Q&A
- **Agentic Orchestrator** (Phase 3): Tool execution for actions

## Components Created

### 1. Intent Classifier (`app/services/intent_classifier.py`)
- LLM-based semantic classification
- Heuristic fallback (no LLM required)
- Detects action verbs, external systems, multi-step workflows

### 2. Query Router (`app/services/query_router.py`)
- Dispatches to RAG or Orchestrator based on intent
- Unified response format
- Configurable confidence threshold

### 3. Tool Manifests (`app/tooling/manifests/`)
- `crm_tool.yaml`: CRM operations (create lead, create ticket, etc.)
- `email_tool.yaml`: Email operations (send transactional, bulk, status)

### 4. Integration (`app/api/endpoints/chat.py`)
- Seamless integration into chat endpoint
- Backward compatible with existing flows

## Setup Instructions

### Step 1: Register Tool Manifests

```bash
cd backend

# Register tools
python -m app.tooling.register_tools
```

This will:
- Load YAML manifests from `app/tooling/manifests/`
- Register them in database (tool_registry tables)
- Display registered operations

### Step 2: Enable Routing for an Agent

Update agent configuration in database or via API:

```python
# Option 1: Enable intelligent routing (RAG + Agentic auto-detection)
agent.config = {
    "enable_intelligent_routing": True,
    "enable_agentic_tools": True,  # Allow tool execution
    "permissions": ["crm.create_lead", "email.send_transactional"]
}

# Option 2: RAG only (no tool execution)
agent.config = {
    "enable_intelligent_routing": True,
    "enable_agentic_tools": False  # Router will only use RAG
}

# Option 3: Disabled (use original flow)
agent.config = {
    "enable_intelligent_routing": False
}
```

### Step 3: Test the Router

#### Test RAG Routing (Information Query)
```bash
curl -X POST http://localhost:8000/api/v1/chat/{agent_public_id} \
  -H "Authorization: Bearer {agent_api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your pricing plans?",
    "conversation_id": "test_conv_123"
  }'

# Response will show:
# - routing_decision: "rag"
# - reasoning: "Information retrieval query pattern detected"
# - tools_used: []
```

#### Test Agentic Routing (Action Query)
```bash
curl -X POST http://localhost:8000/api/v1/chat/{agent_public_id} \
  -H "Authorization: Bearer {agent_api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a CRM ticket for order 00234 - wrong size issue",
    "conversation_id": "test_conv_124"
  }'

# Response will show:
# - routing_decision: "agentic"
# - reasoning: "Detected 1 action verbs and 1 external systems"
# - tools_used: ["crm"]
# - step_results with ticket_id
```

## Configuration Options

### Agent Config Schema
```json
{
  "enable_intelligent_routing": true,        // Enable router (default: false)
  "enable_agentic_tools": true,             // Allow tool execution (default: false)
  "routing_confidence_threshold": 0.7,      // Min confidence for agentic (default: 0.7)
  "permissions": [                          // Tool permissions for RBAC
    "crm.create_lead",
    "crm.create_ticket",
    "email.send_transactional"
  ]
}
```

### Environment Variables
```bash
# No additional env vars required!
# Router uses existing:
# - GEMINI_API_KEY (for LLM classification)
# - DATABASE_URL (for tool registry)
```

## How It Works

### Flow for Information Query
```
User: "What is your refund policy?"
  ↓
IntentClassifier: QueryIntent.RAG (confidence: 0.85)
  ↓
QueryRouter → RAG Service
  ↓
Response: "Our refund policy allows..." (sources: [policy.pdf])
```

### Flow for Action Query
```
User: "Create a Salesforce lead for John Doe from Acme Corp"
  ↓
IntentClassifier: QueryIntent.AGENTIC (confidence: 0.92)
  ↓
QueryRouter → Agent Orchestrator
  ↓
Planner: Generates plan with salesforce.create_lead step
  ↓
Policy Engine: Checks permissions ✓
  ↓
Executor: Calls Salesforce API
  ↓
Response: "✓ Created lead ID: 00Q5e000001YxZ9EAK"
```

## Testing

Run tests:
```bash
cd backend
pytest tests/test_query_router.py -v
```

Test coverage:
- ✅ Intent classification (RAG vs Agentic)
- ✅ Action verb detection
- ✅ External system detection
- ✅ Multi-step workflow detection
- ✅ Router initialization
- ✅ Permission extraction

## Classification Examples

### RAG (Information Retrieval)
- "What are your pricing plans?"
- "How does the product work?"
- "Tell me about enterprise features"
- "Explain the refund policy"
- "Where can I find documentation?"

### AGENTIC (Tool Execution)
- "Create a Salesforce lead for John Doe"
- "Send him a welcome email"
- "Create a support ticket for order 12345"
- "Update CRM ticket status to resolved"
- "Schedule a demo for tomorrow"
- "Create lead and then send email" (multi-step)

## Monitoring

Router adds metadata to responses:
```json
{
  "customer_context": {
    "routing": {
      "decision": "agentic",
      "reasoning": "Detected 2 action verbs and 1 external systems",
      "tools_used": ["crm", "email"],
      "execution_time_ms": 612
    }
  }
}
```

## Troubleshooting

### Router Falls Back to RAG
**Symptom**: Action queries go to RAG instead of Agentic

**Possible causes**:
1. `enable_agentic_tools: false` in agent config
2. Confidence below threshold (default 0.7)
3. No tools registered
4. Permission denied

**Solution**:
```python
# Check agent config
agent.config.get("enable_agentic_tools")  # Should be True

# Check registered tools
await tool_registry.list_tools()  # Should show crm, email

# Check logs
# Look for "Router error" or "Policy gate decision: deny"
```

### Tool Execution Fails
**Symptom**: Router selects agentic but execution fails

**Possible causes**:
1. Tool not registered
2. Permission denied (RBAC)
3. Invalid args schema
4. API endpoint unreachable

**Solution**:
```bash
# Re-register tools
python -m app.tooling.register_tools

# Check permissions in agent config
# Add required permission: "crm.create_ticket"

# Check policy logs in database
SELECT * FROM audit_logs WHERE decision = 'deny';
```

## Next Steps

1. **Add Real API Integrations**
   - Replace mock endpoints with real Salesforce/SendGrid APIs
   - Add authentication secrets to secret store

2. **Add More Tools**
   - Calendar (Google Calendar, Outlook)
   - Payment (Stripe, PayPal)
   - Project Management (Jira, Asana)

3. **Enhance Classification**
   - Fine-tune LLM prompts for better accuracy
   - Add domain-specific training data
   - Implement confidence calibration

4. **Add Monitoring**
   - Track routing decisions (RAG vs Agentic %)
   - Monitor tool execution success rates
   - Alert on policy denials

## API Reference

See:
- `app/services/intent_classifier.py` - Classification logic
- `app/services/query_router.py` - Routing logic
- `app/tooling/` - Tool registry and manifests
- `app/api/endpoints/chat.py:227-362` - Integration point
