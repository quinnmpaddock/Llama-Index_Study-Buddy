#!/usr/bin/env bash
# Run eval_prompts.py with proper NixOS LD_LIBRARY_PATH for numpy/llama_index.
# On non-NixOS systems, this simply delegates to Python directly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build LD_LIBRARY_PATH for NixOS (libstdc++, libz)
NIX_LIB_PATHS=""
if [ -e /etc/NIXOS ]; then
    GCC_LIB="$(nix eval --raw nixpkgs#stdenv.cc.cc.lib 2>/dev/null || true)/lib"
    ZLIB_LIB="$(nix eval --raw nixpkgs#zlib 2>/dev/null || true)/lib"
    [ -d "$GCC_LIB" ] && NIX_LIB_PATHS="$GCC_LIB"
    [ -d "$ZLIB_LIB" ] && NIX_LIB_PATHS="$NIX_LIB_PATHS:$ZLIB_LIB"
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"

if [ -n "$NIX_LIB_PATHS" ]; then
    export LD_LIBRARY_PATH="$NIX_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/eval_prompts.py" "$@"