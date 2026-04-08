#!/usr/bin/env bash
#
# Study Buddy Entrypoint Script
# Starts Neo4j (Docker), Python backend, and launches Rust CLI
#
# Usage: ./study-buddy.sh [--backend-only | --cli-only | --help]
#

set -e

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Neo4j container settings
NEO4J_CONTAINER="neo4j-apoc-gds"
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687
NEO4J_AUTH="neo4j/neo4j2026"
NEO4J_IMAGE="neo4j:latest"

# Backend settings
BACKEND_PORT=8000
BACKEND_HOST="0.0.0.0"

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper Functions ---

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_port() {
    local port=$1
    local service=$2
    
    if lsof -i:"$port" >/dev/null 2>&1; then
        local pid=$(lsof -t -i:"$port" 2>/dev/null | head -1)
        local process=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
        log_error "Port $port is already in use by process: $process (PID: $pid)"
        log_error "Please stop the conflicting process before running Study Buddy."
        return 1
    fi
    return 0
}

check_all_ports() {
    local has_error=0
    
    if ! check_port "$NEO4J_HTTP_PORT" "Neo4j HTTP"; then
        has_error=1
    fi
    
    if ! check_port "$NEO4J_BOLT_PORT" "Neo4j Bolt"; then
        has_error=1
    fi
    
    if ! check_port "$BACKEND_PORT" "Python Backend"; then
        has_error=1
    fi
    
    if [ $has_error -eq 1 ]; then
        log_error "One or more required ports are in use.Aborting."
        exit 1
    fi
    
    log_success "All required ports are available."
}

check_docker() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker is not installed or not in PATH."
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running or you don't have permission."
        echo "Try: sudo systemctl start docker"
        echo "Or add your user to the docker group: sudo usermod -aG docker \$USER"
        exit 1
    fi
    
    log_success "Docker is available and running."
}

start_neo4j() {
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER}$"; then
        # Container exists, check if running
        if docker ps --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER}$"; then
            log_info "Neo4j container is already running."
        else
            log_info "Starting existing Neo4j container..."
            docker start "$NEO4J_CONTAINER" >/dev/null
            log_success "Neo4j container started."
        fi
    else
        log_info "Creating new Neo4j container..."
        docker run -d \
            -p ${NEO4J_HTTP_PORT}:7474 \
            -p ${NEO4J_BOLT_PORT}:7687 \
            -v "${SCRIPT_DIR}/data:/data" \
            -v "${SCRIPT_DIR}/plugins:/plugins" \
            --name "$NEO4J_CONTAINER" \
            -e "NEO4J_AUTH=${NEO4J_AUTH}" \
            -e "NEO4J_apoc_export_file_enabled=true" \
            -e "NEO4J_apoc_import_file_enabled=true" \
            -e "NEO4J_apoc_import_file_use__neo4j__config=true" \
            -e 'NEO4JLABS_PLUGINS=["apoc","graph-data-science"]' \
            "$NEO4J_IMAGE" >/dev/null
        
        log_success "Neo4j container created and started."
    fi
    
    # Wait for Neo4j to be ready
    wait_for_neo4j
}

wait_for_neo4j() {
    log_info "Waiting for Neo4j to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:${NEO4J_HTTP_PORT}" >/dev/null 2>&1; then
            log_success "Neo4j is ready."
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    
    log_error "Neo4j failed to start within ${max_attempts} seconds."
    exit 1
}

check_python_venv() {
    if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
        log_error "Python virtual environment not found at ${SCRIPT_DIR}/.venv"
        echo "Please run: nix-shell (if using Nix) or:"
        echo "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements_backup.txt"
        exit 1
    fi
    log_success "Python virtual environment found."
}

check_rust_binary() {
    local binary_path="${SCRIPT_DIR}/cli/target/release/sb"
    
    if [ ! -f "$binary_path" ]; then
        log_warn "Rust binary not found at $binary_path"
        log_info "Building Rust CLI..."
        cd "${SCRIPT_DIR}/cli"
        cargo build --release 2>&1
        cd "$SCRIPT_DIR"
        
        if [ -f "$binary_path" ]; then
            log_success "Rust CLI built successfully."
        else
            log_error "Failed to build Rust CLI."
            exit 1
        fi
    else
        log_success "Rust CLI binary found."
    fi
}

start_backend() {
    log_info "Starting Python backend..."
    
    cd "$SCRIPT_DIR"
    
    # Source the virtual environment
    source .venv/bin/activate
    
    # Load environment variables from .env
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    
    # Start backend in background (app.py is in src/)
    cd src
    python -m uvicorn app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
    BACKEND_PID=$!
    cd "$SCRIPT_DIR"
    
    log_info "Backend started (PID: $BACKEND_PID)"
    
    # Wait for backend to be ready
    wait_for_backend
}

wait_for_backend() {
    log_info "Waiting for backend to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:${BACKEND_PORT}/" >/dev/null 2>&1; then
            log_success "Backend is ready."
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    
    log_warn "Backend health check timed out, but proceeding anyway..."
}

run_cli() {
    log_info "Launching Study Buddy CLI..."
    
    cd "$SCRIPT_DIR"
    
    # Run the Rust CLI
    "${SCRIPT_DIR}/cli/target/release/sb" "$@"
    
    local exit_code=$?
    cleanup
    exit $exit_code
}

cleanup() {
    log_info "Cleaning up..."
    
    # Kill backend if we started it
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    
    # Note: We don't stop Neo4j - users may want it to persist
    log_success "Cleanup complete."
}

# --- Signal Handlers ---
trap cleanup EXIT

# --- Main ---
main() {
    local mode="full"
    
    # Parse arguments
    case "$1" in
        --backend-only)
            mode="backend"
            ;;
        --cli-only)
            mode="cli";;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend-only    Start Neo4j and Python backend only (no CLI)"
            echo "  --cli-only        Run CLI only (assumes services are running)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Default: Start all services and run CLI"
            exit 0
            ;;
    esac
    
    log_info "Study Buddy - Starting up..."
    
    # Pre-flight checks
    check_docker
    
    if [ "$mode" == "full" ] || [ "$mode" == "backend" ]; then
        check_all_ports
        start_neo4j
        check_python_venv
        start_backend
    fi
    
    if [ "$mode" == "cli" ]; then
        check_rust_binary
        check_python_venv
    fi
    
    if [ "$mode" == "full" ] || [ "$mode" == "cli" ]; then
        check_rust_binary
        run_cli "$@"
    fi
    
    if [ "$mode" == "backend" ]; then
        log_success "Backend is running. Press Ctrl+C to stop."
        wait
    fi
}

main "$@"