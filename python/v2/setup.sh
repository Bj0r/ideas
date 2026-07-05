#!/bin/bash
# setup.sh — runs before pip install on Streamlit Cloud
# Also useful for Xubuntu local fresh-install bootstrap

set -e

echo ">> Installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    pkg-config \
    sqlite3

echo ">> System dependencies installed."
