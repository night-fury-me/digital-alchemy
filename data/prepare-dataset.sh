#!/bin/bash

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running download.py..."
python "$SCRIPT_DIR/download.py"

if [ $? -eq 0 ]; then
    echo "download.py completed successfully."
else
    echo "Error: download.py failed. Exiting."
    exit 1
fi

echo "Running create-db.py..."
python "$SCRIPT_DIR/create-db.py"

if [ $? -eq 0 ]; then
    echo "create-db.py completed successfully."
else
    echo "Error: create-db.py failed. Exiting."
    exit 1
fi

echo "Setup completed successfully!"
