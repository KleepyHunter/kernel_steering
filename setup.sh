#!/bin/bash

# venv alternative to `conda env create -f environment.yml`. Run with `source setup.sh`.
# Needs Python 3.11 as `python3` (the current environment's version; see requirements.txt).
# Run from the repository root.

REQUIRED_PYTHON="3.11"

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python3 not be found."
        return 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [ "$PYTHON_VERSION" != "$REQUIRED_PYTHON" ]; then
        echo "This script requires Python $REQUIRED_PYTHON but found $PYTHON_VERSION"
        return 1
    fi

    return 0
}

setup_venv() {
    echo "Setting up venv..."

    python3 -m venv venv
    source venv/bin/activate

    echo "Done setting up venv!"
}

install_requirements() {
    echo "Installing requirements..."

    pip install -r requirements.txt
    # vLLM and the rest of the current environment, without dependency resolution
    # (see requirements.txt).
    pip install --no-deps -r constraints.txt

    echo "Done installing requirements!"
}

setup_hf() {
    echo "Please enter your Hugging Face token (press Enter to skip):"
    read -r token
    if [ -n "$token" ]; then
        echo "Storing HF_TOKEN in .env file..."
        echo "HF_TOKEN=$token" >> .env

        echo "Logging in to Hugging Face CLI..."
        huggingface-cli login --token $token
    else
        echo "No token entered. Skipping..."
    fi
}

echo "Running set up..."

echo "" > .env

check_python
if [ $? -ne 0 ]; then
    return 1
fi

setup_venv
install_requirements
setup_hf

echo "All set up!"
