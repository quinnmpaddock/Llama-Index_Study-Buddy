#!/usr/bin/env bash
#
# Study Buddy Entrypoint Script
# Starts Neo4j (Docker), Python backend, and launches Rust CLI
#
# Usage: ./study-buddy-server.sh [--backend-only | --cli-only | --help]
#

# Don't exit on error - we handle errors explicitly
# set -e

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
    
    # Try ss first (more reliable), fallback to lsof
    if command -v ss &>/dev/null; then
        if ss -tln 2>/dev/null | grep -q ":${port} "; then
            log_error "Port $port ($service) is already in use."
            log_error "Run: ss -tln | grep :${port} to identify the process."
            return 1
        fi
    elif command -v lsof &>/dev/null; then
        if lsof -i:"$port" >/dev/null 2>&1; then
            log_error "Port $port ($service) is already in use."
            log_error "Run: lsof -i :${port} to identify the process."
            return 1
        fi
    else
        # Fallback: try to bind to the port briefly
        if python3 -c "import socket; s=socket.socket(); s.bind(('', $port)); s.close()" 2>/dev/null; then
            : # Port is free
        else
            log_error "Port $port ($service) appears to be in use."
            return 1
        fi
    fi
    return 0
}

check_all_ports() {
    local has_error=0
    
    # Check if Neo4j container is already running - if so, skip its port checks
    local neo4j_running=false
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${NEO4J_CONTAINER}$"; then
        neo4j_running=true
        log_info "Neo4j container is already running - skipping port checks for Neo4j."
    fi
    
    # Only check Neo4j ports if container isn't running
    if [ "$neo4j_running" = false ]; then
        if ! check_port "$NEO4J_HTTP_PORT" "Neo4j HTTP"; then
            has_error=1
        fi
        
        if ! check_port "$NEO4J_BOLT_PORT" "Neo4j Bolt"; then
            has_error=1
        fi
    fi
    
    if ! check_port "$BACKEND_PORT" "Python Backend"; then
        has_error=1
    fi
    
    if [ $has_error -eq 1 ]; then
        log_error "One or more required ports are in use. Aborting."
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
        echo ""
        echo "Creating virtual environment..."
        python3 -m venv "${SCRIPT_DIR}/.venv"
        
        if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
            log_error "Failed to create virtual environment."
            echo "Please run: python3 -m venv .venv"
            exit 1
        fi
        log_success "Virtual environment created."
    fi
    log_success "Python virtual environment found."
    
    # Source the venv
    source "${SCRIPT_DIR}/.venv/bin/activate"
    
    # Check if we're in a nix-shell (dependencies provided by Nix)
    if [ -n "$IN_NIX_SHELL" ]; then
        log_success "Running inside nix-shell."
        # Nix flake provides numpy, pandas, etc but not all deps - check and install
        if ! python3 -c "from llama_index.node_parser.docling import DoclingNodeParser" 2>/dev/null; then
            log_info "Installing Python dependencies..."
            # Use minimal requirements to let pip resolve compatible versions
            pip install -q -r "${SCRIPT_DIR}/requirements_minimal.txt" 2>&1 | tail -5
            log_success "Dependencies installed."
        fi
        return 0
    fi
    
    # Detect NixOS - venv won't work without nix-shell due to missing libraries
    if [ -f "/etc/NIXOS" ]; then
        log_warn "Running on NixOS without nix-shell."
        log_warn "Virtual environments on NixOS require nix-shell for C libraries."
        echo ""
        if [ -f "${SCRIPT_DIR}/flake.nix" ]; then
            echo "Please run inside nix develop:"
            echo "  nix develop"
            echo "  ./study-buddy.sh"
        else
            echo "Please run inside nix-shell:"
            echo "  nix-shell"
            echo "  ./study-buddy.sh"
        fi
        exit 1
    fi
    
    # Check if core dependencies are installed
    if ! python3 -c "import numpy; import llama_index" 2>/dev/null; then
        log_warn "Core dependencies not installed."
        log_info "Installing dependencies from requirements.txt..."
        log_info "This may take a few minutes on first run..."
        
        # Use requirements.txt (without broken setuptools pin)
        local req_file="${SCRIPT_DIR}/requirements.txt"
        if [ ! -f "$req_file" ]; then
            # Fallback: create from requirements_backup.txt without setuptools pin
            grep -v "^setuptools==" "${SCRIPT_DIR}/requirements_backup.txt" > "$req_file" 2>/dev/null
        fi
        
        if pip install -r "$req_file" 2>&1; then
            log_success "Dependencies installed successfully."
        else
            # Check if at least core deps are available now
            if python3 -c "import numpy; import llama_index" 2>/dev/null; then
                log_success "Core dependencies installed (some optional packages may have failed)."
            else
                log_error "Failed to install core dependencies."
                echo ""
                echo "Try running inside nix-shell:"
                echo "  nix-shell"
                echo "  ./study-buddy.sh"
                exit 1
            fi
        fi
    fi
}

start_backend() {
    log_info "Starting Python backend..."
    
    cd "$SCRIPT_DIR" || exit 1
    
    # Source the virtual environment
    source .venv/bin/activate
    
    # Load environment variables from .env
    # Load environment variables from .env
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi
    
    # Start backend in background (app.py is in src/)
    cd src || exit 1
    python -m uvicorn app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
    BACKEND_PID=$!
    cd "$SCRIPT_DIR" || exit 1
    
    log_info "Backend started (PID: $BACKEND_PID)"
    
    # Wait for backend to be ready
    wait_for_backend
}

wait_for_backend() {
    log_info "Waiting for backend to be ready..."
    log_info "(First run may take 1-2 minutes to download embedding models)"
    local max_attempts=120
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:${BACKEND_PORT}/" >/dev/null 2>&1; then
            log_success "Backend is ready."
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
        
        # Progress indicator every 10 seconds
        if [ $((attempt % 10)) -eq 0 ]; then
            log_info "Still waiting... ($attempt seconds)"
        fi
    done
    
    # Health check failed - backend might not be responding
    log_error "Backend health check failed after ${max_attempts} seconds."
    log_error "Checking if process is still running..."
    
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        log_error "Process $BACKEND_PID is running but not responding on port $BACKEND_PORT."
        log_error "This may indicate a startup hang (e.g., Neo4j connection issue)."
        log_error "Check the Python logs above for errors."
    else
        log_error "Process $BACKEND_PID has crashed."
    fi
    
    exit 1
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
    # Parse arguments
    case "$1" in
        --help|-h)
            echo "Study Buddy Server - Starts Neo4j and Python backend"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "This script starts the backend services (Neo4j + Python API)."
            echo "Use the 'sb' CLI tool to interact with the running backend."
            echo ""
            echo "Examples:"
            echo "  $0                          # Start backend services"
            echo "  ./sb query 'your question'  # Query the knowledge graph"
            echo "  ./sb ingest input/          # Ingest documents"
            echo "  ./sb tui                    # Interactive TUI mode"
            exit 0
            ;;
    esac
    
    log_info "Study Buddy Server - Starting up..."
    
    # Pre-flight checks
    check_docker
    check_all_ports
    start_neo4j
    check_python_venv
    start_backend
    
    log_success "Backend is running. Press Ctrl+C to stop."
    wait
}

main "$@"