#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-"https://api.yourdomain.com"}
AGENT_PUBLIC_ID=${AGENT_PUBLIC_ID:-"replace-with-agent-public-id"}
AGENT_API_KEY=${AGENT_API_KEY:-"replace-with-agent-api-key"}
AGENT_NUMERIC_ID=${AGENT_NUMERIC_ID:-42}
ADMIN_BEARER=${ADMIN_BEARER:-""}

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

ingest_document() {
  echo "Ingesting Shopify article via admin API..."
  curl -s "https://katanamrp.com/integrations/shopify/shopify-order-management/" \
    | jq -Rs '{filename: "shopify-order-management.html", content: ., content_type: "text/html"}' \
    | curl -s -X POST "$API_URL/api/v1/documents/agent/$AGENT_NUMERIC_ID" \
        -H "Authorization: Bearer $ADMIN_BEARER" \
        -H "Content-Type: application/json" \
        -d @-

  echo "Document ingestion request sent (processing is async)."
}

get_session_token() {
  curl -s -X POST \
    "$API_URL/api/v1/agents/public/$AGENT_PUBLIC_ID/session-token" \
    -H "X-Agent-API-Key: $AGENT_API_KEY" \
    | jq -r '.token'
}

stream_question() {
  local token="$1"
  curl -N "$API_URL/api/v1/chat/$AGENT_PUBLIC_ID/stream" \
    -H "Accept: text/event-stream" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $token" \
    -d @- <<'JSON'
{
  "message": "According to https://katanamrp.com/integrations/shopify/shopify-order-management/, what are the Shopify order lifecycle steps and how does Katana sync them?",
  "user_id": "curl-demo-user",
  "session_context": {
    "page_url": "https://katanamrp.com/integrations/shopify/shopify-order-management/",
    "referrer": "curl-demo"
  }
}
JSON
}

main() {
  API_URL=$(prompt_for_value "API_URL" "$API_URL")
  AGENT_PUBLIC_ID=$(prompt_for_value "AGENT_PUBLIC_ID" "$AGENT_PUBLIC_ID")
  AGENT_API_KEY=$(prompt_for_value "AGENT_API_KEY" "$AGENT_API_KEY")
  AGENT_NUMERIC_ID=$(prompt_for_value "AGENT_NUMERIC_ID" "$AGENT_NUMERIC_ID")

  echo "\n--- RAG CURL DEMO ---"
  echo "API_URL              : $API_URL"
  echo "AGENT_PUBLIC_ID      : $AGENT_PUBLIC_ID"
  echo "AGENT_NUMERIC_ID     : $AGENT_NUMERIC_ID"

  read -rp "Ingest Shopify article via admin API? [y/N]: " answer
  answer=${answer:-N}
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    ADMIN_BEARER=$(prompt_for_value "ADMIN_BEARER" "$ADMIN_BEARER")
    if [[ -z "$ADMIN_BEARER" || "$ADMIN_BEARER" == "replace-with"* ]]; then
      echo "ERROR: Admin bearer token required to ingest documents." >&2
      exit 1
    fi
    ingest_document
  else
    echo "Skipping ingestion."
  fi

  echo "\nRequesting short-lived session token..."
  SESSION_TOKEN=$(get_session_token)
  if [[ -z "$SESSION_TOKEN" || "$SESSION_TOKEN" == "null" ]]; then
    echo "ERROR: Failed to obtain session token. Check public ID / API key." >&2
    exit 1
  fi
  echo "Session token acquired.\n"

  echo "Streaming demo question (SSE)...\n"
  stream_question "$SESSION_TOKEN"

  echo "\n--- Demo complete ---"
}

main "$@"
