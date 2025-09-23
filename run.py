"""
Main entry point for the Multi-Agent AHP Experiment System

This file provides a clean entry point for running the application
from the root directory while maintaining the organized src/ structure.
"""

from src.app import app

if __name__ == '__main__':
    # Bind to 0.0.0.0 so the container port is reachable from host
    app.run(debug=True, host='0.0.0.0', port=5002)
