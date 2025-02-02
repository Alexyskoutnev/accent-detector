#!/bin/bash

set -e

# Create virtual environment
VENV_NAME=venv
python3 -m venv $VENV_NAME
# Activate virtual environment
source $VENV_NAME/bin/activate
# You might need to manually activate the virtual environment
echo "You might need to manually activate the virtual environment"
echo "Run: source $VENV_NAME/bin/activate"
# Install requirements
pip install -r requirements.txt
