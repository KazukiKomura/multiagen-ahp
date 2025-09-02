"""
Multi-Agent AHP Experiment System - Main Application

This is the main Flask application that orchestrates all components of the 
procedural justice-enhanced multi-agent decision-making system.

Architecture:
- services/: Business logic (ProceduralJusticeSystem)
- repository/: Data access layer (SessionRepository)  
- routes/: HTTP endpoints (main, ai_chat)
- utils/: Utility functions (data processing)
"""

from flask import Flask
import os

# Import blueprints
from .routes.main import main_bp
from .routes.ai_chat import ai_chat_bp

# Import repository to ensure database initialization
from .repository.session_repository import session_repository


def create_app():
    """Application factory pattern"""
    # Configure Flask to use src/templates and src/static
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    app.secret_key = 'experiment-secret-key-2025'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(ai_chat_bp)
    
    # Initialize database
    session_repository._init_db()
    
    return app


# Create application instance
app = create_app()


if __name__ == '__main__':
    app.run(debug=True, port=5000)