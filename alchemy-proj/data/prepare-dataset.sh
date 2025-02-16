#!/bin/bash

set -e

echo "Running download.py..."
python download.py

if [ $? -eq 0 ]; then
    echo "download.py completed successfully."
else
    echo "Error: download.py failed. Exiting."
    exit 1
fi

echo "Running create-db.py..."
python create-db.py

if [ $? -eq 0 ]; then
    echo "create-db.py completed successfully."
else
    echo "Error: create-db.py failed. Exiting."
    exit 1
fi

echo "Setup completed successfully!"