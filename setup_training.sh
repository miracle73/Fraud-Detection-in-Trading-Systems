#!/bin/bash
echo "🚀 Setting up ML Training Environment..."

# Install Python packages
pip install -r requirements_ml.txt

# Install additional system packages if needed
sudo apt-get update -qq
sudo apt-get install -y screen htop

# Create directories
mkdir -p trained_models
mkdir -p training_logs
mkdir -p results

echo "✅ Training environment setup complete!"