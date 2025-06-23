#!/bin/bash

set -e

RUNS=${1:-30}

echo ">>> Krok 1: Budowanie obrazu Docker 'cma_experiments'..."
docker build -t cma_experiments .

echo ""
echo ">>> Krok 2: Uruchamianie eksperymentu (${RUNS} przebiegów)..."
docker run \
    --rm \
    -it \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/plots:/app/plots" \
    cma_experiments "$RUNS"

echo ""
echo ">>> Skrypt zakończył działanie. Wyniki w folderach 'data' i 'plots'."
