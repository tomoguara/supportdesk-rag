#!/usr/bin/env bash
#
# qdrant-db.sh - Start/Stop Qdrant Vector Database
#
# Usage:
#   ./qdrant-db.sh --start    Start Qdrant container
#   ./qdrant-db.sh --stop     Stop and clean up Qdrant container
#   ./qdrant-db.sh --status   Check if Qdrant is running
#   ./qdrant-db.sh --wait     Wait for Qdrant to be ready (max 30 seconds)
#

set -e

# Configuration
QDRANT_CONTAINER_NAME="supportdesk-qdrant"
QDRANT_PORT="6333"
QDRANT_IMAGE="qdrant/qdrant"
WAIT_TIMEOUT=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
}

# Start Qdrant
start_qdrant() {
    check_docker
    
    # Check if already running
    if docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        log_info "Qdrant is already running"
        log_info "Container: ${QDRANT_CONTAINER_NAME}"
        log_info "URL: http://localhost:${QDRANT_PORT}"
        return 0
    fi
    
    # Check if stopped container exists
    if docker ps --all --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        log_info "Stopping and removing existing container..."
        docker rm -f ${QDRANT_CONTAINER_NAME} || true
    fi
    
    log_info "Starting Qdrant container..."
    log_info "Container name: ${QDRANT_CONTAINER_NAME}"
    log_info "Port: ${QDRANT_PORT}"
    log_info "Image: ${QDRANT_IMAGE}"
    
    docker run \
        --name ${QDRANT_CONTAINER_NAME} \
        -p ${QDRANT_PORT}:6333 \
        -d \
        ${QDRANT_IMAGE}
    
    log_info "Qdrant started"
    log_info "Waiting for container to be ready..."
    
    wait_for_qdrant
}

# Stop Qdrant
stop_qdrant() {
    check_docker
    
    if ! docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        log_info "Qdrant is not running"
        return 0
    fi
    
    log_info "Stopping Qdrant container..."
    docker stop ${QDRANT_CONTAINER_NAME}
    
    log_info "Removing Qdrant container..."
    docker rm ${QDRANT_CONTAINER_NAME}
    
    log_info "Qdrant stopped and removed"
}

# Check status
status_qdrant() {
    check_docker
    
    if docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        log_info "Qdrant is RUNNING"
        echo "  Container: ${QDRANT_CONTAINER_NAME}"
        echo "  URL: http://localhost:${QDRANT_PORT}"
        echo "  Status: $(docker inspect --format='{{.State.Status}}' ${QDRANT_CONTAINER_NAME})"
        
        # Test connection
        if curl -s http://localhost:${QDRANT_PORT} > /dev/null 2>&1; then
            log_info "Connection test: PASSED"
            curl -s http://localhost:${QDRANT_PORT} | head -1 | sed 's/^/  /'
        else
            log_warn "Connection test: FAILED (container running but not responding)"
        fi
        return 0
    elif docker ps --all --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        log_warn "Qdrant container exists but is STOPPED"
        echo "  Container: ${QDRANT_CONTAINER_NAME}"
        echo "  Status: stopped"
        return 1
    else
        log_info "Qdrant is NOT RUNNING"
        return 0
    fi
}

# Wait for Qdrant to be ready
wait_for_qdrant() {
    log_info "Waiting for Qdrant to be ready (timeout: ${WAIT_TIMEOUT}s)..."
    
    local elapsed=0
    while [ $elapsed -lt $WAIT_TIMEOUT ]; do
        if curl -s http://localhost:${QDRANT_PORT} > /dev/null 2>&1; then
            log_info "Qdrant is ready!"
            return 0
        fi
        
        # Sleep 1 second and increment
        sleep 1
        elapsed=$((elapsed + 1))
        
        # Progress indicator
        local progress=""
        for ((i=0; i<elapsed && i<10; i++)); do
            progress+="#"
        done
        for ((i=elapsed; i<10; i++)); do
            progress+="."
        done
        echo -ne "\r  [${progress}] ${elapsed}s"
    done
    
    echo ""
    log_error "Qdrant failed to start within ${WAIT_TIMEOUT} seconds"
    log_error "Try: ./qdrant-db.sh --status to check the container"
    return 1
}

# Show help
show_help() {
    echo "Qdrant Database Management Script"
    echo ""
    echo "Usage: ./qdrant-db.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  --start    Start Qdrant container"
    echo "  --stop     Stop and remove Qdrant container"
    echo "  --status   Check Qdrant status"
    echo "  --wait     Wait for Qdrant to be ready (use after --start)"
    echo "  --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./qdrant-db.sh --start    # Start Qdrant"
    echo "  ./qdrant-db.sh --stop     # Stop Qdrant"
    echo "  ./qdrant-db.sh --status   # Check if Qdrant is running"
    echo ""
    echo "Configuration:"
    echo "  Container Name: ${QDRANT_CONTAINER_NAME}"
    echo "  Port: ${QDRANT_PORT}"
    echo "  Image: ${QDRANT_IMAGE}"
}

# Main
case "${1:-}" in
    --start)
        start_qdrant
        ;;
    --stop)
        stop_qdrant
        ;;
    --status)
        status_qdrant
        ;;
    --wait)
        wait_for_qdrant
        ;;
    --help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: ${1:-}"
        echo ""
        show_help
        exit 1
        ;;
esac
