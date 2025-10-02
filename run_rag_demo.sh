#!/usr/bin/env bash
set -euo pipefail

API_URL="https://api.yourdomain.com"
AGENT_PUBLIC_ID="replace-with-agent-public-id"
AGENT_API_KEY="replace-with-agent-api-key"
AGENT_NUMERIC_ID=42
ADMIN_BEARER="replace-with-admin-console-jwt-if-needed"

prompt_for_value() {
  local var_name="$1"
  local current_value="$2"

  if [[ -z "$current_value" || "$current_value" == "replace-with"* ]]; then
    read -rp "Enter value for $var_name: " new_value
    echo "$new_value"
  else
    echo "$current_value"
  fi
}

main() {
  API_URL=$(prompt_for_value "API_URL" "$API_URL")
  AGENT_PUBLIC_ID=$(prompt_for_value "AGENT_PUBLIC_ID" "$AGENT_PUBLIC_ID")
  AGENT_API_KEY=$(prompt_for_value "AGENT_API_KEY" "$AGENT_API_KEY")
  AGENT_NUMERIC_ID=$(prompt_for_value "AGENT_NUMERIC_ID" "$AGENT_NUMERIC_ID")
  ADMIN_BEARER=$(prompt_for_value "ADMIN_BEARER (press Enter if not ingesting docs)" "$ADMIN_BEARER")

  echo "\n--- RAG Demo ---"
  echo "API_URL=$API_URL"
  echo "AGENT_PUBLIC_ID=$AGENT_PUBLIC_ID"
  echo "AGENT_NUMERIC_ID=$AGENT_NUMERIC_ID"

  read -rp "Do you want to ingest https://katanamrp.com/integrations/shopify/shopify-order-management/ into the agent documents? [y/N]: " ingest_answer
  ingest_answer=${ingest_answer:-N}

  if [[ "$ingest_answer" =~ ^[Yy]$ ]]; then
    if [[ -z "$ADMIN_BEARER" || "$ADMIN_BEARER" == "replace-with-admin"* ]]; then
      echo "\nERROR: ADMIN_BEARER token is required to ingest documents." >&2
      exit 1
    fi

    echo "\nIngesting Shopify order-management article..."
    curl -s "https://katanamrp.com/integrations/shopify/shopify-order-management/" \
      | jq -Rs '{filename: "shopify-order-management.html", content: ., content_type: "text/html"}' \
      | curl -s -X POST "$API_URL/api/v1/documents/agent/$AGENT_NUMERIC_ID" \
          -H "Authorization: Bearer $ADMIN_BEARER" \
          -H "Content-Type: application/json" \
          -d @-

    echo "Document ingestion request sent."
  else
    echo "Skipping ingestion."
  fi

  echo "\nRequesting session token..."
  SESSION_TOKEN=$(curl -s -X POST \
    "$API_URL/api/v1/agents/public/$AGENT_PUBLIC_ID/session-token" \
    -H "X-Agent-API-Key: $AGENT_API_KEY" \
    | jq -r '.token')

  if [[ "$SESSION_TOKEN" == "null" || -z "$SESSION_TOKEN" ]]; then
    echo "\nERROR: Failed to obtain session token. Check credentials." >&2
    exit 1
  fi

  echo "Session token acquired.\n"

  echo "Streaming question via SSE...\n"
  curl -N "$API_URL/api/v1/chat/$AGENT_PUBLIC_ID/stream" \
    -H "Accept: text/event-stream" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $SESSION_TOKEN" \
    -d @- <<EOF2
{
  "message": "According to https://katanamrp.com/integrations/shopify/shopify-order-management/, what are the Shopify order lifecycle steps and how does Katana sync them?",
  "user_id": "curl-demo-user",
  "session_context": {
    "page_url": "https://katanamrp.com/integrations/shopify/shopify-order-management/",
    "referrer": "curl-demo"
  }
}
EOF2

  echo "\n--- Demo complete ---"
}

main "$@"
