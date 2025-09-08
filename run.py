"""
Main entry point for the Multi-Agent AHP Experiment System

This file provides a clean entry point for running the application
from the root directory while maintaining the organized src/ structure.
"""

from src.app import app

if __name__ == '__main__':
    app.run(debug=True, port=5002)
