#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/tiny_shakespeare"

echo "Creating data directory..."
mkdir -p "$DATA_DIR"

echo "Downloading tiny-shakespeare dataset..."
curl -o "$DATA_DIR/input.txt" https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

echo "Download complete! Data saved to $DATA_DIR/input.txt"
