#!/bin/bash

# Wikipedia Maturity Dashboard Startup Script

echo "🚀 Starting Wikipedia Maturity Dashboard..."
echo ""

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "❌ Virtual environment not found. Please run from project root with venv activated."
    exit 1
fi

# Activate virtual environment
source ../venv/bin/activate

# Install Flask dependencies if not already installed
echo "📦 Installing Flask dependencies..."
pip install -q Flask==3.0.0 Flask-CORS==4.0.0

echo ""
echo "🌐 Starting dashboard server..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo "🔧 API endpoints:"
echo "   • GET /api/article/<title> - Get article maturity score"
echo "   • GET /api/peers/<title> - Get peer articles"
echo "   • GET /api/search?q=<query> - Search articles"
echo "   • GET /api/health - Health check"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask server
python app.py
