#!/usr/bin/env bash
# LLMObsidian launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting ollama server..."
    ollama serve &
    sleep 2
fi

# Optionally override vault path
if [ -n "$1" ]; then
    export VAULT_PATH="$1"
    echo "Using vault: $VAULT_PATH"
fi

echo "Starting LLMObsidian..."
streamlit run app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
