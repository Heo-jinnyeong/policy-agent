#!/usr/bin/env bash
# Download only missing Policy Agent models. Existing models and servers are kept.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v ollama >/dev/null 2>&1 || { echo "ERROR: ollama가 필요합니다."; exit 1; }
if [ -f .env ]; then set -a; source .env; set +a; fi
QUALITY_MODEL="${OLLAMA_QUALITY_MODEL:-qwen3:8b}"
EMBEDDING_MODEL="${POLICY_EMBEDDING_MODEL:-qwen3-embedding:0.6b}"
STARTED_PID=""

if ! curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    ollama serve > /tmp/policy_agent_ollama_setup.log 2>&1 &
    STARTED_PID=$!
    for _ in $(seq 1 30); do
        curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
        sleep 0.5
    done
fi

cleanup() {
    [ -n "$STARTED_PID" ] && kill "$STARTED_PID" 2>/dev/null || true
}
trap cleanup EXIT

curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
    || { echo "ERROR: Ollama 서버가 응답하지 않습니다."; exit 1; }

for model in "$QUALITY_MODEL" "$EMBEDDING_MODEL"; do
    if ollama list | grep "^${model}[[:space:]]" >/dev/null; then
        echo "기존 모델 재사용: $model"
    else
        echo "모델 다운로드: $model"
        ollama pull "$model"
    fi
done

echo "Policy Agent 모델 준비 완료: $QUALITY_MODEL / $EMBEDDING_MODEL"
