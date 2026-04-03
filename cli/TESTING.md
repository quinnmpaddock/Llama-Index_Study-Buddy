# Testing the Study Buddy CLI

## Option 1: Mock API Server (Quick Testing)

The mock server simulates API responses without requiring Neo4j or the full backend.

```bash
# Terminal 1: Start the mock server
cd cli
python mock_api.py

# Terminal 2: Test the CLI
./target/release/sb status
./target/release/sb query "What are knowledge graphs?"
./target/release/sb query "Explain fuzzing" -f json
./target/release/sb query "Tell me about testing" -t 30
```

## Option 2: Full Backend (Integration Testing)

Requires Neo4j database running.

### Prerequisites

1. **Start Neo4j**:
```bash
# Using Docker:
docker run -d \
  --name neo4j \
  -p7474:7474 -p7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:latest

# Or using nix-shell/service:
# systemctl start neo4j  # or your system's service manager
```

2. **Activate Python environment**:
```bash
cd ..  # back to study-buddy root
source .venv/bin/activate  # or: .venv/bin/activate
```

3. **Set environment variables** (from .env):
```bash
export OPENAI_API_KEY="your-key-here"
# The app.py reads NEO4JPASSWORD from the file directly
```

4. **Start FastAPI server**:
```bash
cd src
python -m uvicorn app:app --reload --port 8000
```

5. **Test the CLI**:
```bash
cd ../cli
./target/release/sb query "What is knowledge graph synthesis?"
```

## Option 3: Unit Tests

Run Rust unit tests:

```bash
cd cli
cargo test
```

## Testing Checklist

- [ ] `sb status` - Shows connection status
- [ ] `sb config` - Shows configuration
- [ ] `sb query "test"` - Basic query
- [ ] `sb query "test" -f json` - JSON output
- [ ] `sb query "test" -f yaml` - YAML output
- [ ] `sb query "test" -t 50` - Custom top-k
- [ ] `sb --no-color query "test"` - No color output
- [ ] `sb -u http://other:8000 query "test"` - Custom URL