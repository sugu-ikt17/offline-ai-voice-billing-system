#!/usr/bin/env bash
# =============================================================================
# setup_whisper.sh — Build whisper.cpp and download the tiny.en model.
#
# Usage:
#   chmod +x scripts/setup_whisper.sh
#   ./scripts/setup_whisper.sh
#
# What it does:
#   1. Clones whisper.cpp into a temporary build directory.
#   2. Compiles the `main` binary (requires gcc / g++ and cmake or make).
#   3. Copies the compiled binary to  models/whisper/main
#   4. Downloads ggml-tiny.en.bin to  models/whisper/ggml-tiny.en.bin
#
# Environment variable overrides (optional):
#   WHISPER_BINARY_PATH  — custom destination for the compiled binary
#   WHISPER_MODEL_PATH   — custom destination for the model file
#
# Tested on: Ubuntu 22.04 / Debian 11 / Raspberry Pi OS (64-bit)
# =============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/models/whisper"

BINARY_DEST="${WHISPER_BINARY_PATH:-${MODELS_DIR}/main}"
MODEL_DEST="${WHISPER_MODEL_PATH:-${MODELS_DIR}/ggml-tiny.en.bin}"

WHISPER_REPO="https://github.com/ggerganov/whisper.cpp.git"
WHISPER_TAG="v1.7.4"   # pinned for reproducibility
BUILD_DIR="/tmp/whisper-cpp-build-$$"

# ─── Helpers ──────────────────────────────────────────────────────────────────
info()  { echo "[setup_whisper] INFO : $*"; }
warn()  { echo "[setup_whisper] WARN : $*" >&2; }
error() { echo "[setup_whisper] ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || error "'$1' is required but not found. Install it and retry."
}

# ─── Pre-flight checks ────────────────────────────────────────────────────────
info "Checking prerequisites..."
require_cmd git
require_cmd make
require_cmd g++

mkdir -p "$MODELS_DIR"
info "Models directory: $MODELS_DIR"

# ─── Skip if already built ────────────────────────────────────────────────────
if [[ -x "$BINARY_DEST" && -f "$MODEL_DEST" ]]; then
    info "whisper.cpp binary and model already present — nothing to do."
    info "  Binary : $BINARY_DEST"
    info "  Model  : $MODEL_DEST"
    exit 0
fi

# ─── Clone & build whisper.cpp ────────────────────────────────────────────────
if [[ ! -x "$BINARY_DEST" ]]; then
    info "Cloning whisper.cpp @ $WHISPER_TAG..."
    git clone --depth 1 --branch "$WHISPER_TAG" "$WHISPER_REPO" "$BUILD_DIR"

    info "Building whisper.cpp (this may take a few minutes on Raspberry Pi)..."
    make -C "$BUILD_DIR" -j"$(nproc)" main

    info "Copying binary to $BINARY_DEST"
    cp "$BUILD_DIR/main" "$BINARY_DEST"
    chmod +x "$BINARY_DEST"

    info "Cleaning up build directory..."
    rm -rf "$BUILD_DIR"
else
    info "Binary already exists at $BINARY_DEST — skipping build."
fi

# ─── Download tiny.en model ───────────────────────────────────────────────────
if [[ ! -f "$MODEL_DEST" ]]; then
    info "Downloading ggml-tiny.en.bin (~39 MB)..."

    # Official Hugging Face mirror — reliable, no login required
    MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"

    if command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar -o "$MODEL_DEST" "$MODEL_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$MODEL_DEST" "$MODEL_URL"
    else
        error "Neither curl nor wget found. Please install one and retry."
    fi

    info "Model downloaded: $MODEL_DEST"
else
    info "Model already exists at $MODEL_DEST — skipping download."
fi

# ─── Verify ───────────────────────────────────────────────────────────────────
info "Verifying installation..."
[[ -x "$BINARY_DEST" ]] || error "Binary not found or not executable: $BINARY_DEST"
[[ -f "$MODEL_DEST"   ]] || error "Model file not found: $MODEL_DEST"

info "✓ whisper.cpp binary : $BINARY_DEST"
info "✓ Model (tiny.en)    : $MODEL_DEST"
info ""
info "Setup complete! The speech engine is ready."
info ""
info "Optional — set custom paths in your .env file:"
info "  WHISPER_BINARY_PATH=$BINARY_DEST"
info "  WHISPER_MODEL_PATH=$MODEL_DEST"
