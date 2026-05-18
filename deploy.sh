#!/bin/bash

##############################################################################
# NIM + IPEX-LLM Deployment Orchestrator
# Usage: ./deploy.sh {setup|start|stop|restart|logs|health|test|config|help}
##############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# FIX 1: All bash functions were missing their closing } braces — script
# would fail to parse entirely. All braces restored below.

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

##############################################################################
# Configuration & Auto-Detection
##############################################################################

# Auto-detect active compose file and backend name
if [ -f "docker-compose-llm-scaler.yml" ]; then
    COMPOSE_FILE="docker-compose-llm-scaler.yml"
    BACKEND_SERVICE="llm-scaler-backend"
    BACKEND_NAME="Intel LLM-Scaler"
else
    COMPOSE_FILE="docker-compose.yml"
    BACKEND_SERVICE="ipex-backend"
    BACKEND_NAME="IPEX-LLM"
fi

# Helper function to call docker-compose with the right file
call_compose() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

load_config() {
    if [ ! -f ".env" ]; then
        log_error ".env file not found"
        log_info "Creating template .env file..."
        cat > .env << 'EOF'
# ===== REQUIRED INPUTS =====
NIM_CONTAINER_URL=nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
LLM_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
NGC_API_KEY=your_actual_ngc_api_key_here

# ===== OPTIONAL (defaults provided) =====
IPEX_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest
NIM_PROXY_PORT=8000
IPEX_BACKEND_PORT=8001
NIM_CACHE_PATH=${HOME}/.cache/nim
HF_CACHE_PATH=${HOME}/.cache/huggingface
DOCKER_NETWORK=nim-ipex-net
EOF
        log_info "Edit .env with your values and run again"
        exit 1
    fi

    export $(grep -v '^#' .env | grep -v '^$' | xargs)

    for var in NIM_CONTAINER_URL LLM_MODEL_NAME NGC_API_KEY; do
        if [ -z "${!var}" ]; then
            log_error "$var is not set in .env"
            exit 1
        fi
    done

    log_success "Configuration loaded"
}

show_config() {
    echo ""
    echo -e "${BLUE}Current Configuration:${NC}"
    echo "  NIM Container:     $NIM_CONTAINER_URL"
    echo "  LLM Model:         $LLM_MODEL_NAME"
    echo "  NGC API Key:       $(echo $NGC_API_KEY | cut -c1-8)****"
    echo "  IPEX Image:        ${IPEX_IMAGE:-intelanalytics/ipex-llm-serving-xpu:latest}"
    echo "  NIM Proxy Port:    ${NIM_PROXY_PORT:-8000}"
    echo "  IPEX Backend Port: ${IPEX_BACKEND_PORT:-8001}"
    echo "  NIM Cache:         ${NIM_CACHE_PATH:-~/.cache/nim}"
    echo "  HF Cache:          ${HF_CACHE_PATH:-~/.cache/huggingface}"
    echo ""
}

##############################################################################
# Commands
##############################################################################

cmd_setup() {
    log_info "Setting up directories..."
    mkdir -p "${NIM_CACHE_PATH:-$HOME/.cache/nim}"
    mkdir -p "${HF_CACHE_PATH:-$HOME/.cache/huggingface}"
    mkdir -p ./config
    mkdir -p ./models

    if [ ! -f "config/nim-adapter.py" ]; then
        log_info "Copying nim-adapter.py to config/"
        cp ./nim-adapter.py ./config/ 2>/dev/null || \
            log_warn "nim-adapter.py not found — place it at ./config/nim-adapter.py"
    fi

    chmod 777 "${NIM_CACHE_PATH:-$HOME/.cache/nim}" \
              "${HF_CACHE_PATH:-$HOME/.cache/huggingface}" 2>/dev/null || true
    log_success "Setup complete"
}

cmd_start() {
    log_info "Starting NIM + ${BACKEND_NAME} stack..."
    cmd_setup
    call_compose up -d
    log_info "Initial startup may take 2-5 min for SYCL kernel compilation."
    log_info "Watch progress: docker-compose -f ${COMPOSE_FILE} logs -f ${BACKEND_SERVICE}"
    echo ""
    log_success "Stack started"
}

cmd_stop() {
    log_info "Stopping NIM + ${BACKEND_NAME} stack..."
    call_compose down
    log_success "Stack stopped"
}

cmd_restart() {
    cmd_stop
    sleep 3
    cmd_start
}

cmd_logs() {
    log_info "Showing logs (Ctrl+C to exit)..."
    call_compose logs -f ${BACKEND_SERVICE} nim-proxy
}

cmd_health() {
    echo ""
    echo -e "${BLUE}Service Health Check${NC}"
    echo ""

    echo -n "${BACKEND_NAME} Backend (port ${IPEX_BACKEND_PORT:-8001}): "
    if curl -sf "http://localhost:${IPEX_BACKEND_PORT:-8001}/v1/models" > /dev/null 2>&1; then
        log_success "responding"
    else
        log_error "not responding"
    fi

    echo -n "NIM Proxy (port ${NIM_PROXY_PORT:-8000}): "
    if curl -sf "http://localhost:${NIM_PROXY_PORT:-8000}/v1/models" > /dev/null 2>&1; then
        log_success "responding"
    else
        log_error "not responding"
    fi

    echo ""
    log_info "Container status:"
    call_compose ps --no-trunc || log_warn "Docker compose not initialized"
    echo ""
}

cmd_test() {
    log_info "Running inference test..."
    echo ""

    response=$(curl -s "http://localhost:${NIM_PROXY_PORT:-8000}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$LLM_MODEL_NAME\",
            \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer briefly.\"}],
            \"max_tokens\": 50,
            \"temperature\": 0
        }")

    if echo "$response" | jq . > /dev/null 2>&1; then
        log_success "Inference completed"
        echo ""
        echo -e "${BLUE}Response:${NC}"
        echo "$response" | jq '.choices[0].message.content' || echo "$response" | jq .
    else
        log_error "Inference failed"
        echo ""
        echo -e "${BLUE}Raw response:${NC}"
        echo "$response"
    fi
    echo ""
}

cmd_config() {
    show_config
    echo -e "${BLUE}To change configuration:${NC}"
    echo "  Edit .env, then run: ./deploy.sh restart"
    echo ""
}

cmd_help() {
    cat << 'EOF'
NIM + IPEX-LLM Deployment Orchestrator

Usage: ./deploy.sh {command}

Commands:
  setup     Create directories and copy adapter file
  start     Start the full stack
  stop      Stop the stack
  restart   Restart (use after editing .env)
  logs      Stream live logs from both services
  health    Check service health
  test      Run an inference test
  config    Show current configuration
  help      Show this help

First-time Setup:
  1. Edit .env  (NIM_CONTAINER_URL, LLM_MODEL_NAME, NGC_API_KEY)
  2. ./deploy.sh setup
  3. ./deploy.sh start
  4. ./deploy.sh logs     (watch kernel compilation)
  5. ./deploy.sh test

See NIM_IPEX_LLM_Intel_iGPU_Deployment_Guide.md for full details.
EOF
}

##############################################################################
# Main
##############################################################################

main() {
    local cmd=${1:-help}
    case "$cmd" in
        setup)        load_config; cmd_setup ;;
        start)        load_config; show_config; cmd_start ;;
        stop)         load_config; cmd_stop ;;
        restart)      load_config; cmd_restart ;;
        logs)         load_config; cmd_logs ;;
        health)       load_config; cmd_health ;;
        test)         load_config; cmd_test ;;
        config)       load_config; cmd_config ;;
        help|-h|--help) cmd_help ;;
        *)
            log_error "Unknown command: $cmd"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
